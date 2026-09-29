"""Evaluation harness: runs seeded episodes per agent type, applies the
detectors, and computes detection metrics.

Metrics per hack type (ensemble vs. honest baseline):
- recall     = fraction of that hack type's episodes flagged
- precision  = fraction of flagged episodes that are actual hacks (global)
- F1         = harmonic mean of the two
- honest FPR = fraction of honest episodes wrongly flagged
"""

from __future__ import annotations

from .agents import AGENT_TYPES, HACK_TYPES, run_agent
from .detectors import EnsembleDetector


def evaluate(n_episodes: int = 200, seed: int = 7,
             divergence_weight: float = 0.5) -> dict:
    detector = EnsembleDetector(divergence_weight=divergence_weight)
    flagged_by_type: dict[str, int] = {t: 0 for t in AGENT_TYPES}
    totals: dict[str, int] = {t: 0 for t in AGENT_TYPES}

    for i, agent_type in enumerate(AGENT_TYPES):
        for j in range(n_episodes):
            ep_seed = seed * 1_000_003 + i * 10_007 + j
            ep = run_agent(agent_type, ep_seed)
            if detector.verdict(ep).flagged:
                flagged_by_type[agent_type] += 1
            totals[agent_type] += 1

    total_flagged = sum(flagged_by_type.values())
    hacks_flagged = sum(flagged_by_type[t] for t in HACK_TYPES)
    total_hacks = sum(totals[t] for t in HACK_TYPES)
    precision = hacks_flagged / total_flagged if total_flagged else 0.0

    per_type: dict[str, dict[str, float]] = {}
    for t in HACK_TYPES:
        recall = flagged_by_type[t] / totals[t]
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_type[t] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "flagged": flagged_by_type[t],
            "total": totals[t],
        }

    honest_fpr = flagged_by_type["honest"] / totals["honest"]
    return {
        "n_episodes_per_type": n_episodes,
        "seed": seed,
        "per_type": per_type,
        "honest_false_positive_rate": honest_fpr,
        "honest_flagged": flagged_by_type["honest"],
        "honest_total": totals["honest"],
    }
