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
   intervals, in immediate-answer and reasoning modes, against:
   - controls built exactly like the PTS vector but without the information
     under test: shuffled signs, and random positions in the same traces;
   - CAA, a direction from the prompt and answer alone;
   - a null of random vectors of the same size at the same setting;
   - plain prompting ("Answer in a way that avoids {behavior}.").
   Controls are tuned on the selection split exactly like the PTS vector.

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
from .evals import load_choice_eval, split
from .scoring import ChoiceScorer
from .search import SearchConfig, search
from .validate import Evaluator, bootstrap_ci, choose, paired_effect, random_like, single_layer

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    num_items: int = 120
    seed: int = 0
    layers: Optional[List[int]] = None              # default: a band through the middle
    fractions: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    min_mass_ratio: float = 0.9
    max_kl: float = 0.5
    random_null: int = 10                           # random vectors per chosen setting
    cot_samples: int = 16                           # 0 skips the reasoning-mode evaluation
    prompt_increase: Optional[str] = None           # default: "Answer in a way that shows {behavior}."
    prompt_decrease: Optional[str] = None
    reuse_search: bool = False                      # load events/traces already in output_dir
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


def _run_search(scorer, disc, config, model_id, behavior, output_dir):
    from ..event_storage import EventStorage

    events_path = os.path.join(output_dir, "events.jsonl")
    traces_path = os.path.join(output_dir, "traces.jsonl")
    if config.reuse_search and os.path.exists(events_path) and os.path.exists(traces_path):
        events = list(EventStorage(filepath=events_path))
        with open(traces_path) as f:
            traces = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Reusing {len(events)} events and {len(traces)} traces from {output_dir}")
        return events, traces, None
    storage = EventStorage(filepath=events_path)
    traces: List[dict] = []
    events = search(scorer, disc, config.search, model_id, behavior, storage=storage, traces_out=traces)
    storage.save()
    with open(traces_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return events, traces, scorer.cost.as_dict()


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
    mx.random.seed(config.seed)
    scorer = ChoiceScorer(model, tokenizer)
    hidden, num_layers = mlx_backend.model_shape(model)
    layers = config.layers or default_layers(num_layers)
    capture = lambda ids, ls: mlx_backend.capture(model, ids, ls)

    items = load_choice_eval(eval_path, limit=config.num_items, seed=config.seed)
    disc, sel, test = split(items)
    logger.info(f"{behavior}: {len(disc)} discovery / {len(sel)} selection / {len(test)} test questions")

    events, traces, search_cost = _run_search(scorer, disc, config, model_id, behavior, output_dir)

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

    evaluator = Evaluator(scorer, model_id=model_id)
    base_sel, base_test = evaluator.score(sel), evaluator.score(test)
    base_test_cot = evaluator.score(test, cot_samples=config.cot_samples) if config.cot_samples else None
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
        "layers": layers, "fractions": config.fractions,
        "base": {"selection": base_sel.as_dict(), "test": base_test.as_dict(),
                 "test_cot": base_test_cot.as_dict() if base_test_cot else None},
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
                    s = evaluator.score(sel, [(one, alpha)])
                    rows.append({"layer": layer, "fraction": sign * frac, "alpha": alpha, **s.as_dict()})
        method = {"count": vec.provenance["count"], "selection_sweep": rows}
        for direction, label in ((1, "increase"), (-1, "decrease")):
            best = choose(rows, base_sel, direction, config.min_mass_ratio, config.max_kl)
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
                "selection": {k: best[k] for k in ("p_match", "mass", "kl")},
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
            if base_test_cot is not None:
                cot = evaluator.score(test, steering, cot_samples=config.cot_samples)
                result["test_cot"] = cot.as_dict()
                result["test_cot_effect"] = paired_effect(cot, base_test_cot, seed=config.seed)
                chosen_scores.setdefault(label, {})[name + ":cot"] = cot
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
        direct = evaluator.score(test, prompt=text)
        entry = {"prompt": text, "test": direct.as_dict(), "test_effect": paired_effect(direct, base_test)}
        chosen_scores.setdefault(label, {})["prompt"] = direct
        if base_test_cot is not None:
            cot = evaluator.score(test, prompt=text, cot_samples=config.cot_samples)
            entry["test_cot_effect"] = paired_effect(cot, base_test_cot)
            chosen_scores[label]["prompt:cot"] = cot
        report["prompt_baselines"][label] = entry

    # Paired: is the PTS vector better than each alternative on the same questions?
    report["comparisons"] = {}
    for label, direction in (("increase", 1), ("decrease", -1)):
        scores = chosen_scores.get(label, {})
        for mode in ("", ":cot"):
            main = scores.get(MAIN + mode)
            if main is None:
                continue
            for other, s in scores.items():
                if other.endswith(":cot") != bool(mode) or other == MAIN + mode:
                    continue
                diffs = direction * (np.array(main.per_item) - np.array(s.per_item))
                lo, hi = bootstrap_ci(diffs, seed=config.seed)
                report["comparisons"][f"{label}{mode}: {MAIN} vs {other.replace(mode, '')}"] = {
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
            f"mass {r['test']['mass']:.2f}, KL {r['test']['kl']:.3f}")
