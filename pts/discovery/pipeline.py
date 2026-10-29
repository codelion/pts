"""
End-to-end behavior discovery: eval -> pivotal tokens -> vectors -> validated steering.

1. Split the eval by question into discovery / selection / test.
2. Search the discovery questions for pivotal tokens (``search``).
3. Build candidate directions from those events, plus the CAA baseline from
   the same discovery questions.
4. For each method, sweep layer and strength on the selection questions and
   pick the strongest setting that passes the damage checks, separately for
   increasing and decreasing the behavior.
5. Report those settings on the test questions with paired bootstrap
   intervals against:
   - controls built exactly like the PTS vector but without the information
     under test: shuffled signs, and random positions in the same traces;
   - CAA, a direction from the prompt and answer alone;
   - a null of random vectors of the same size at the same setting;
   - plain prompting ("Answer in a way that avoids {behavior}.").
   Controls are tuned on the selection split exactly like the PTS vector.

Every P(behavior), in the search and in validation, is token PTS's estimator:
sample completions of the dataset's own prompt and ask the oracle whether
each succeeded. The selection sweep uses fewer samples per question than the
test evaluation; it only has to rank settings.

Strength is relative: ``fraction`` is the size of the push as a fraction of
the layer's typical residual norm. Methods produce vectors of very different
raw norms, so comparing them at equal alpha would compare unequal pushes.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .directions import (
    caa,
    matched_pairs,
    random_position_pairs,
    sign_shuffled_pairs,
    signed_mean,
    to_vector,
)
from .evals import ChoiceOracle, load_choice_eval, split
from .scoring import SAMPLING_KEYS, ChoiceScorer, limit_gpu_memory
from .search import SearchConfig, search_item
from .validate import Evaluator, bootstrap_ci, choose, paired_effect, random_like, single_layer

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    num_items: int = 120
    seed: int = 0
    layers: Optional[List[int]] = None              # default: a band through the middle
    fractions: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    min_answered_ratio: float = 0.9
    max_kl: float = 0.5
    eval_samples: int = 16                          # completions per test question
    selection_samples: int = 4                      # completions per selection question, per setting
    random_null: int = 5                            # random vectors per chosen setting
    prompt_increase: Optional[str] = None           # default: "Answer in a way that shows {behavior}."
    prompt_decrease: Optional[str] = None
    reuse_search: bool = False                      # load events/traces already in output_dir
    chat_template_args: Dict = field(default_factory=dict)   # e.g. {"enable_thinking": False}
    memory_fraction: float = 0.5                    # MLX memory cap, as a fraction of device memory
    sampling: Dict = field(default_factory=dict)    # ChoiceScorer sampling, e.g. a model card's settings
    search: SearchConfig = field(default_factory=SearchConfig)


# The method under test, then everything it has to beat.
MAIN = "pts_matched_pairs"


def default_layers(num_layers: int) -> List[int]:
    lo, hi = int(round(0.3 * num_layers)), int(round(0.75 * num_layers))
    return list(range(lo, min(hi, num_layers - 1) + 1, 2))


def _alpha(vec, layer: int, fraction: float) -> float:
    e = next(e for e in vec.entries if e.layer == layer)
    norm = float(np.linalg.norm(e.vector))
    return fraction * e.ref_norm / norm if norm > 0 else 0.0


def _run_search(scorer, disc, config, model_id, eval_path, behavior, output_dir):
    """Search the discovery questions, saving after each one.

    A rerun with the same settings resumes where the last one stopped. With
    ``reuse_search`` the saved events and traces are used as they are, even if
    the settings differ (e.g. to re-validate on a larger test split).
    """
    from tqdm import tqdm

    from ..event_storage import EventStorage

    events_path = os.path.join(output_dir, "events.jsonl")
    traces_path = os.path.join(output_dir, "traces.jsonl")
    state_path = os.path.join(output_dir, "search_state.json")

    def load_traces():
        if not os.path.exists(traces_path):
            return []
        with open(traces_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    if config.reuse_search and os.path.exists(events_path):
        events, traces = list(EventStorage(filepath=events_path)), load_traces()
        logger.info(f"Reusing {len(events)} events and {len(traces)} traces from {output_dir}")
        return events, traces, None

    settings = {
        "model": model_id, "eval": eval_path, "seed": config.seed,
        "items": [it.item_id for it in disc], "search": asdict(config.search),
        "sampling": {k: getattr(scorer, k) for k in SAMPLING_KEYS},
        "chat_template_args": scorer.chat_template_args,
    }
    state = {"settings": settings, "done": [], "cost": {}}
    if os.path.exists(state_path):
        with open(state_path) as f:
            saved = json.load(f)
        if saved["settings"] != settings:
            raise ValueError(
                f"{output_dir} holds a search with different settings; use a new --output-dir "
                "or --reuse-search to use its events as they are"
            )
        state = saved
    else:
        for path in (events_path, traces_path):
            if os.path.exists(path):
                os.remove(path)

    storage = EventStorage(filepath=events_path)
    done = set(state["done"])
    if done:
        logger.info(f"Resuming search: {len(done)}/{len(disc)} questions already done")
    prior = dict(state["cost"])
    todo = [it for it in disc if it.item_id not in done]
    for item in tqdm(todo, desc=f"Searching {behavior}", initial=len(done), total=len(disc)):
        traces: List[dict] = []
        found = list(search_item(scorer, item, config.search, model_id, behavior, traces))
        storage.add_events(found)
        storage.save()
        with open(traces_path, "a") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        state["done"].append(item.item_id)
        now = scorer.cost.as_dict()
        state["cost"] = {k: prior.get(k, 0) + v for k, v in now.items()}
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(state_path + ".tmp", state_path)
        logger.info(f"{item.item_id}: {len(found)} pivotal tokens, {len(traces)} moved traces "
                    f"({len(state['done'])}/{len(disc)} questions, {len(storage)} events)")
    return list(storage), load_traces(), state["cost"]


def discover(
    model,
    tokenizer,
    model_id: str,
    eval_path: str,
    behavior: str,
    output_dir: str,
    config: DiscoveryConfig = DiscoveryConfig(),
) -> Dict:
    import mlx.core as mx

    from ..steering import mlx_backend

    os.makedirs(output_dir, exist_ok=True)
    limit_gpu_memory(config.memory_fraction)
    mx.random.seed(config.seed)
    items = load_choice_eval(eval_path, limit=config.num_items, seed=config.seed)
    disc, sel, test = split(items)
    logger.info(f"{behavior}: {len(disc)} discovery / {len(sel)} selection / {len(test)} test questions")

    scorer = ChoiceScorer(model, tokenizer, ChoiceOracle(items),
                          chat_template_args=config.chat_template_args, **config.sampling)
    hidden, num_layers = mlx_backend.model_shape(model)
    layers = config.layers or default_layers(num_layers)
    capture = lambda ids, ls: mlx_backend.capture(model, ids, ls)

    events, traces, search_cost = _run_search(scorer, disc, config, model_id, eval_path, behavior, output_dir)

    builders = {
        MAIN: lambda: matched_pairs(scorer, capture, events, layers),
        "pts_signed_mean_centered": lambda: signed_mean(scorer, capture, events, layers),
        "control_sign_shuffle": lambda: sign_shuffled_pairs(scorer, capture, events, layers, config.seed),
        "control_random_positions": lambda: random_position_pairs(
            scorer, capture, events, traces, layers, config.seed),
        "caa": lambda: caa(scorer, capture, disc, layers),
    }
    candidates = {}
    for name, build in builders.items():
        try:
            vecs, norms, count = build()
        except ValueError as e:
            logger.warning(f"{name}: {e}")
            continue
        candidates[name] = to_vector(vecs, norms, hidden, num_layers, name, count, model_id, behavior)

    max_tokens = config.search.max_tokens
    selector = Evaluator(scorer, model_id, samples=config.selection_samples, max_tokens=max_tokens)
    evaluator = Evaluator(scorer, model_id, samples=config.eval_samples, max_tokens=max_tokens)
    base_sel, base_test = selector.score(sel), evaluator.score(test)
    report = {
        "behavior": behavior, "model": model_id, "eval": eval_path,
        "split": {"discovery": len(disc), "selection": len(sel), "test": len(test)},
        "events": {
            "total": len(events),
            "positive": sum(e.prob_delta > 0 for e in events),
            "states_answer": sum(bool(e.metadata.get("states_answer")) for e in events),
            "moved_traces": len(traces),
        },
        "search_config": asdict(config.search), "search_cost": search_cost,
        "chat_template_args": config.chat_template_args, "sampling": config.sampling,
        "eval_samples": config.eval_samples, "selection_samples": config.selection_samples,
        "layers": layers, "fractions": config.fractions,
        "base": {"selection": base_sel.as_dict(), "test": base_test.as_dict()},
        "methods": {},
    }

    chosen_scores: Dict[str, Dict[str, object]] = {}
    for name, vec in candidates.items():
        rows = []
        for layer in layers:
            one = single_layer(vec, layer)
            for frac in config.fractions:
                for sign in (1, -1):
                    alpha = sign * _alpha(vec, layer, frac)
                    s = selector.score(sel, [(one, alpha)])
                    rows.append({"layer": layer, "fraction": sign * frac, "alpha": alpha, **s.as_dict()})
        method = {"count": vec.provenance["count"], "selection_sweep": rows}
        for direction, label in ((1, "increase"), (-1, "decrease")):
            best = choose(rows, base_sel, direction, config.min_answered_ratio, config.max_kl)
            if best is None:
                method[label] = None
                continue
            one = single_layer(vec, best["layer"])
            steering = [(one, best["alpha"])]
            tested = evaluator.score(test, steering)
            null = [
                paired_effect(evaluator.score(test, [(random_like(vec, best["layer"], seed=k), best["alpha"])]),
                              base_test)["delta"]
                for k in range(config.random_null)
            ]
            effect = paired_effect(tested, base_test, seed=config.seed)
            result = {
                "layer": best["layer"], "fraction": best["fraction"], "alpha": best["alpha"],
                "selection": {k: best[k] for k in ("p_match", "answered", "kl")},
                "test": tested.as_dict(),
                "test_effect": effect,
                "random_null": {
                    "deltas": null,
                    "mean": float(np.mean(null)) if null else None,
                    # How often a random vector of the same size, at the same
                    # setting, moved P at least as far in this direction.
                    "p_value": float(np.mean([direction * d >= direction * effect["delta"] for d in null]))
                    if null else None,
                },
            }
            chosen_scores.setdefault(label, {})[name] = tested
            method[label] = result
            one.default_alpha = best["alpha"]
            one.validation = {"behavior": behavior, "direction": label, "eval": eval_path,
                              "base_test": base_test.as_dict(),
                              **{k: v for k, v in result.items() if k != "random_null"},
                              "random_null_p": result["random_null"]["p_value"]}
            one.save(os.path.join(output_dir, f"{name}_{label}"))
        report["methods"][name] = method
        logger.info(f"{name}: increase {_summary(method['increase'])}; decrease {_summary(method['decrease'])}")

    report["prompt_baselines"] = {}
    prompts = {
        "increase": config.prompt_increase or f"Answer in a way that shows {behavior}.",
        "decrease": config.prompt_decrease or f"Answer in a way that avoids {behavior}.",
    }
    for label, text in prompts.items():
        prompted = evaluator.score(test, prompt=text)
        chosen_scores.setdefault(label, {})["prompt"] = prompted
        report["prompt_baselines"][label] = {
            "prompt": text, "test": prompted.as_dict(), "test_effect": paired_effect(prompted, base_test),
        }

    # Paired: is the PTS vector better than each alternative on the same questions?
    report["comparisons"] = {}
    for label, direction in (("increase", 1), ("decrease", -1)):
        scores = chosen_scores.get(label, {})
        main = scores.get(MAIN)
        if main is None:
            continue
        for other, s in scores.items():
            if other == MAIN:
                continue
            diffs = direction * (np.array(main.per_item) - np.array(s.per_item))
            lo, hi = bootstrap_ci(diffs, seed=config.seed)
            report["comparisons"][f"{label}: {MAIN} vs {other}"] = {
                "advantage": float(diffs.mean()), "ci_low": lo, "ci_high": hi,
            }

    report["total_cost"] = scorer.cost.as_dict()
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _summary(r) -> str:
    if r is None:
        return "no setting passed the damage checks"
    e = r["test_effect"]
    return (f"layer {r['layer']} x{r['fraction']:+.2f}: test dP={e['delta']:+.3f} "
            f"[{e['ci_low']:+.3f}, {e['ci_high']:+.3f}], random p={r['random_null']['p_value']}, "
            f"answered {r['test']['answered']:.2f}, KL {r['test']['kl']:.3f}")
