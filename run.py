"""Entry point: run the reward-hack detection experiment.

Runs N seeded episodes per agent type, applies the ensemble detector, prints
a detection-metrics table, and writes results/results.json + a bar-chart PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> dict:
    parser = argparse.ArgumentParser(description="Detect reward hacking in agentic pipelines.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="episodes per agent type (default: 200)")
    parser.add_argument("--seed", type=int, default=7,
                        help="master seed (default: 7)")
    args = parser.parse_args()

    from reward_hack_detector import evaluate

    metrics = evaluate(n_episodes=args.episodes, seed=args.seed)

    print(f"\nReward-hack detection — {metrics['n_episodes_per_type']} episodes/type, seed {metrics['seed']}\n")
    print(f"{'hack type':<16} {'recall':>8} {'precision':>10} {'F1':>8}   flagged")
    print("-" * 58)
    for hack, m in metrics["per_type"].items():
        print(f"{hack:<16} {m['recall']:>8.3f} {m['precision']:>10.3f} "
              f"{m['f1']:>8.3f}   {m['flagged']}/{m['total']}")
    print("-" * 58)
    print(f"honest-agent false-positive rate: {metrics['honest_false_positive_rate']:.3f} "
          f"({metrics['honest_flagged']}/{metrics['honest_total']})\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'results.json'}")

    _plot(metrics)
    return metrics


def _plot(metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(metrics["per_type"].keys())
    recalls = [metrics["per_type"][t]["recall"] for t in labels]
    f1s = [metrics["per_type"][t]["f1"] for t in labels]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.35
    ax.bar([i - w / 2 for i in x], recalls, w, label="recall")
    ax.bar([i + w / 2 for i in x], f1s, w, label="F1")
    ax.axhline(metrics["honest_false_positive_rate"], color="red", linestyle="--",
               label=f"honest FPR ({metrics['honest_false_positive_rate']:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Reward-hack detection: recall / F1 per hack type")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "detection_metrics.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
