"""Token usage and USD accounting with a hard budget cap.

Prices are first-party Anthropic API list prices in USD per million tokens (checked 2026-11).
A 5-minute cache write costs 1.25x the input price; a cache read costs the listed read price.
Costs here are ESTIMATES computed from `response.usage`; the invoice is the source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-opus-5": Price(5.00, 25.00, 0.50),
    "claude-opus-4-8": Price(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-fable-5-1": Price(10.00, 50.00, 0.25),
    "claude-fable-5": Price(10.00, 50.00, 1.00),
}
# Unknown model (e.g. a server-side fallback we have no row for): assume the priciest tier.
UNKNOWN_PRICE = Price(10.00, 50.00, 1.00)


def price_for(*models: str | None) -> Price:
    """The most expensive price among `models`, so a fallback never under-reports spend."""
    ps = [PRICES.get(m or "", UNKNOWN_PRICE) for m in models if m] or [UNKNOWN_PRICE]
    return max(ps, key=lambda p: (p.output, p.input))


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a call that could push spend past the cap."""


@dataclass
class CallUsage:
    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    usd: float = 0.0


@dataclass
class UsageMeter:
    budget_usd: float | None = None
    calls: list[CallUsage] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return sum(c.usd for c in self.calls)

    def totals(self) -> dict[str, float]:
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "usd")
        return {k: sum(getattr(c, k) for c in self.calls) for k in keys} | {"calls": len(self.calls)}

    def check(self, worst_case_usd: float, role: str) -> None:
        """Refuse a call whose worst case (max_tokens of output + estimated input) breaks the cap."""
        if self.budget_usd is None:
            return
        if self.usd + worst_case_usd > self.budget_usd:
            raise BudgetExceeded(
                f"{role} call refused: spent ${self.usd:.4f}, worst case for this call ${worst_case_usd:.4f}, "
                f"budget ${self.budget_usd:.2f}"
            )

    def record(self, role: str, requested_model: str, served_model: str | None, usage: object) -> CallUsage:
        g = lambda k: int(getattr(usage, k, 0) or 0)
        c = CallUsage(
            role=role,
            model=served_model or requested_model,
            input_tokens=g("input_tokens"),
            output_tokens=g("output_tokens"),
            cache_creation_input_tokens=g("cache_creation_input_tokens"),
            cache_read_input_tokens=g("cache_read_input_tokens"),
        )
        p = price_for(requested_model, served_model)
        c.usd = (
            c.input_tokens * p.input
            + c.output_tokens * p.output
            + c.cache_creation_input_tokens * p.cache_write
            + c.cache_read_input_tokens * p.cache_read
        ) / 1e6
        self.calls.append(c)
        return c

    def snapshot(self) -> dict[str, float]:
        return self.totals()

    def as_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def worst_case_usd(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Upper-bound estimate: ~3 chars/token (conservative for code) billed as uncached input."""
    p = price_for(model)
    return (prompt_chars / 3 * p.input + max_tokens * p.output) / 1e6
