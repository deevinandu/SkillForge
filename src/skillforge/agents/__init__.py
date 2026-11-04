"""T3: agents that write and improve skills. Every improver implements contracts.SkillImprover.

Importing this package never imports the Anthropic SDK; `ClaudeImprover` does so on construction.
"""

from .budget import BudgetExceeded, UsageMeter
from .claude_code import ClaudeCodeImprover
from .fake import FakeImprover
from .library import SkillLibrary, passes_gate
from .llm_improver import ClaudeImprover, ImproverError, MissingCredentials
from .loop import LoopResult, RoundRecord, run_loop
from .null import NullImprover
from .sandbox import SandboxResult, check_source, evaluate_source

__all__ = [
    "BudgetExceeded",
    "ClaudeCodeImprover",
    "ClaudeImprover",
    "FakeImprover",
    "ImproverError",
    "LoopResult",
    "MissingCredentials",
    "NullImprover",
    "RoundRecord",
    "SandboxResult",
    "SkillLibrary",
    "UsageMeter",
    "check_source",
    "evaluate_source",
    "passes_gate",
    "run_loop",
]
