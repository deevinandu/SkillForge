"""T4: evaluation, robustness and regression.

Light imports only: `parallel`, `robustness`, `regression`, `video`, `dashboard` are
submodules, imported on use (forge.py imports this package).
"""

from .harness import evaluate, run_episode, summarize, wilson
from .parallel import EvalPool, evaluate_parallel

__all__ = ["EvalPool", "evaluate", "evaluate_parallel", "run_episode", "summarize", "wilson"]
