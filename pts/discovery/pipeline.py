"""
End-to-end behavior discovery: eval -> pivotal tokens -> vectors -> validated steering.

1. Split the eval by question into discovery / selection / test.
2. Search the discovery questions for pivotal tokens (``search``).
3. Build candidate directions from those events, plus the CAA baseline from
   the same discovery questions.
4. For each method, sweep layer and strength on the selection questions and
   pick the strongest setting that passes the damage checks, separately for
   increasing and decreasing the behavior.
5. Report those settings on the test questions, next to a random direction of
   the same size at the same layer and strength.

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

from .directions import caa, matched_pairs, signed_mean, to_vector
from .evals import load_choice_eval, split
from .scoring import ChoiceScorer
from .search import SearchConfig, search
from .validate import Evaluator, choose, random_like, single_layer

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    num_items: int = 120
    seed: int = 0
    layers: Optional[List[int]] = None              # default: a band through the middle
    fractions: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    min_mass_ratio: float = 0.9
    max_kl: float = 0.5
    search: SearchConfig = field(default_factory=SearchConfig)


def default_layers(num_layers: int) -> List[int]:
    lo, hi = int(round(0.3 * num_layers)), int(round(0.75 * num_layers))
    return list(range(lo, min(hi, num_layers - 1) + 1, 2))


def _alpha(vec, layer: int, fraction: float) -> float:
    e = next(e for e in vec.entries if e.layer == layer)
    norm = float(np.linalg.norm(e.vector))
    return fraction * e.ref_norm / norm if norm > 0 else 0.0


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

    from ..event_storage import EventStorage
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

    storage = EventStorage(filepath=os.path.join(output_dir, "events.jsonl"))
    events = search(scorer, disc, config.search, model_id, behavior, storage=storage)
    storage.save()
    search_cost = scorer.cost.as_dict()

    candidates = {}
    for name, build, source in (
        ("pts_matched_pairs", matched_pairs, events),
        ("pts_signed_mean", signed_mean, events),
        ("caa", caa, disc),
    ):
        try:
            vecs, norms, count = build(scorer, capture, source, layers)
        except ValueError as e:
            logger.warning(f"{name}: {e}")
            continue
        candidates[name] = to_vector(vecs, norms, hidden, num_layers, name, count, model_id, behavior)

    evaluator = Evaluator(scorer, model_id=model_id)
    base_sel, base_test = evaluator.score(sel), evaluator.score(test)
    report = {
        "behavior": behavior, "model": model_id, "eval": eval_path,
        "split": {"discovery": len(disc), "selection": len(sel), "test": len(test)},
        "events": {"total": len(events), "positive": sum(e.prob_delta > 0 for e in events)},
        "search_config": asdict(config.search), "search_cost": search_cost,
        "layers": layers, "fractions": config.fractions,
        "base": {"selection": base_sel.as_dict(), "test": base_test.as_dict()},
        "methods": {},
    }

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
            rand = random_like(vec, best["layer"], seed=config.seed)
            tested = evaluator.score(test, [(one, best["alpha"])])
            control = evaluator.score(test, [(rand, best["alpha"])])
            result = {
                "layer": best["layer"], "fraction": best["fraction"], "alpha": best["alpha"],
                "selection": {k: best[k] for k in ("p_match", "mass", "kl")},
                "test": tested.as_dict(),
                "test_delta": tested.p_match - base_test.p_match,
                "random_test": control.as_dict(),
                "random_delta": control.p_match - base_test.p_match,
            }
            method[label] = result
            one.default_alpha = best["alpha"]
            one.validation = {"behavior": behavior, "direction": label, "eval": eval_path,
                              "base_test": base_test.as_dict(), **result}
            one.save(os.path.join(output_dir, f"{name}_{label}"))
        report["methods"][name] = method
        logger.info(
            f"{name}: increase {_summary(method['increase'])}; decrease {_summary(method['decrease'])}"
        )

    report["total_cost"] = scorer.cost.as_dict()
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def _summary(r) -> str:
    if r is None:
        return "no setting passed the damage checks"
    return (f"layer {r['layer']} x{r['fraction']:+.2f}: test dP={r['test_delta']:+.3f} "
            f"(random {r['random_delta']:+.3f}), mass {r['test']['mass']:.2f}, KL {r['test']['kl']:.3f}")
