from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


def normalize_token_count(total_tokens: int | None) -> int:
    if total_tokens is None:
        return 0
    try:
        normalized_total_tokens = int(total_tokens)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported token count value: {total_tokens!r}") from exc
    return max(0, normalized_total_tokens)


@dataclass
class SampleEfficiencyTracker:
    started_at: float = field(default_factory=perf_counter)
    llm_total_tokens: int = 0

    def add_llm_total_tokens(self, total_tokens: int | None) -> None:
        self.llm_total_tokens += normalize_token_count(total_tokens)

    def finalize(self) -> dict[str, float | int]:
        elapsed_seconds = perf_counter() - self.started_at
        return {
            "sample_elapsed_seconds": round(elapsed_seconds, 6),
            "llm_total_tokens": self.llm_total_tokens,
        }


__all__ = ["SampleEfficiencyTracker", "normalize_token_count"]
