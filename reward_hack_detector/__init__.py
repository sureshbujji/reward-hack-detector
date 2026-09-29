"""reward_hack_detector — a harness that detects reward hacking in agentic pipelines.

Simulated end-to-end: a gameable proxy reward, several reward-hacking agent
archetypes, behavioral + divergence detectors, and an evaluation harness that
measures detection precision/recall/F1 per hack type and false-positive rate
on an honest baseline.
"""

from .env import Episode, proxy_reward, ground_truth_quality, independent_audit
from .agents import AGENT_TYPES, run_agent
from .detectors import DivergenceDetector, BehavioralSignatureDetector, EnsembleDetector
from .harness import evaluate

__all__ = [
    "Episode",
    "proxy_reward",
    "ground_truth_quality",
    "independent_audit",
    "AGENT_TYPES",
    "run_agent",
    "DivergenceDetector",
    "BehavioralSignatureDetector",
    "EnsembleDetector",
    "evaluate",
]

__version__ = "0.1.0"
