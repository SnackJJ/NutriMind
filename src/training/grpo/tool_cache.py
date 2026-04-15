"""
Per-prompt-group tool result cache for deterministic GRPO rollouts.

In GRPO, each prompt generates G rollouts. Without caching, the same tool call
(e.g., get_food_nutrition("chicken breast")) could return different results across
rollouts due to retrieval non-determinism, causing reward noise unrelated to policy
quality.

ToolCache ensures identical tool calls within the same prompt group always return
identical results. It also provides snapshot/restore for future ARPO partial rollouts.

Usage:
    cache = ToolCache()

    # Start a new prompt group (clears previous cache)
    cache.new_group("prompt_abc123")

    # Cached tool call — calls real_fn on miss, returns cached on hit
    result = cache.get_or_call("get_food_nutrition", {"foods": [...]}, real_fn)

    # ARPO Phase 2: save/restore at branch points
    snap = cache.snapshot()
    # ... partial rollout ...
    cache.restore(snap)
"""

import json
import logging
from copy import deepcopy
from typing import Any, Callable

logger = logging.getLogger(__name__)


def normalize_value(v: Any) -> Any:
    """Recursively normalize a value for deterministic cache key generation.

    Rules:
        - str: lowercase + strip
        - float: round to 1 decimal (avoids floating-point noise)
        - list: normalize each element (order preserved)
        - dict: normalize values, sort by key
        - other: pass through
    """
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 1)
    if isinstance(v, list):
        return [normalize_value(item) for item in v]
    if isinstance(v, dict):
        return {k: normalize_value(val) for k, val in sorted(v.items())}
    return v


def make_cache_key(tool_name: str, args: dict) -> str:
    """Build a deterministic cache key from tool name and normalized arguments."""
    normalized = normalize_value(args)
    return f"{tool_name}:{json.dumps(normalized, sort_keys=True, ensure_ascii=False)}"


class ToolCache:
    """Thread-unsafe, single-process tool result cache scoped to a prompt group.

    Designed for TRL's environment_factory where rollouts for the same prompt
    run sequentially in the same process.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._current_group: str = ""
        self._hits: int = 0
        self._misses: int = 0

    def new_group(self, group_id: str) -> None:
        """Begin a new prompt group. Clears the cache if the group changed.

        Called by NutriMindToolEnv.reset(). Within the same group (same prompt,
        multiple rollouts), the cache accumulates. When a new prompt arrives,
        the cache is flushed.
        """
        if group_id != self._current_group:
            if self._store:
                logger.debug(
                    "ToolCache: group %s → %s, flushing %d entries (hits=%d, misses=%d)",
                    self._current_group[:8], group_id[:8],
                    len(self._store), self._hits, self._misses,
                )
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._current_group = group_id

    def get_or_call(self, tool_name: str, args: dict, fn: Callable[[], Any]) -> str:
        """Return cached result or call fn(), cache it, and return.

        Args:
            tool_name: Name of the tool (e.g., "get_food_nutrition").
            args: Tool arguments dict.
            fn: Zero-arg callable that executes the real tool and returns
                a dict (will be json.dumps'd) or a str.

        Returns:
            JSON string of the tool result.
        """
        key = make_cache_key(tool_name, args)

        if key in self._store:
            self._hits += 1
            return self._store[key]

        self._misses += 1
        raw_result = fn()

        if isinstance(raw_result, str):
            result_str = raw_result
        else:
            result_str = json.dumps(raw_result, ensure_ascii=False, default=str)

        self._store[key] = result_str
        return result_str

    # ------------------------------------------------------------------
    # ARPO Phase 2: snapshot / restore for partial rollout branching
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Save current cache state. Returns an opaque dict to pass to restore()."""
        return {
            "store": deepcopy(self._store),
            "group": self._current_group,
        }

    def restore(self, snap: dict) -> None:
        """Restore cache state from a previous snapshot."""
        self._store = deepcopy(snap["store"])
        self._current_group = snap["group"]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "group": self._current_group[:16],
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }
