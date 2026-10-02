"""Ticket 016 — compatibility guard (Seam 5), part 1: public borrowed-API imports.

`PUBLIC_BORROWED_API` below is the single source of truth for the whole guard:
it transcribes, verbatim, every symbol of the "Public borrowed API" table in
`.scratch/nutrimind-v2/spec.md` §18. The pinned-signature test
(`test_borrowed_api_signatures.py`) and the two-class-rule test
(`test_two_class_rule.py`) import this table — never duplicate it.

ADR-012 (amended 2026-09-08) two-class rule this guard encodes:

- **Public borrowed API** — symbols in nutri-env's `__all__` that v2 calls
  directly: a formal dependency. Import guard (this file) + pinned-signature
  guard + behaviour tests (elsewhere — Seam 1).
- **Private implementation detail** — the underscore-prefixed helpers of §18's
  private list: no stability promise, not frozen by this guard, covered
  indirectly by the end-to-end behaviour test. `test_two_class_rule.py`
  asserts the guard files stay clean of them.

Name note: an earlier reading of §18's `nutrienv.world.types` row saw
`MAX_ITEM_GRAM`, but the spec and the pinned rev agree on the plural
`MAX_ITEM_GRAMS` (world/types.py: `MAX_ITEM_GRAMS = 2000.0`, in `__all__`) —
no deviation exists; the table below is verbatim.

CI constraint (ticket 016): this file runs in a fresh venv holding ONLY
pytest + nutrienv — it imports nothing beyond nutrienv, the stdlib, pytest,
and sibling test modules in this directory.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

# (module dotted path, symbol name) — spec §18 "Public borrowed API" table,
# verbatim: module by module in table order, symbols in each row's order.
PUBLIC_BORROWED_API: list[tuple[str, str]] = [
    # nutrienv.bench (re-exports)
    ("nutrienv.bench", "Oracle"),
    ("nutrienv.bench", "Task"),
    ("nutrienv.bench", "Scorer"),
    ("nutrienv.bench", "check_achievable"),
    ("nutrienv.bench", "load_split"),
    ("nutrienv.bench", "load_exam"),
    ("nutrienv.bench", "EXAM_SPLIT_PATH"),
    # nutrienv.bench.pipeline.generate_one
    ("nutrienv.bench.pipeline.generate_one", "generate_one"),
    ("nutrienv.bench.pipeline.generate_one", "make_log_expander"),
    ("nutrienv.bench.pipeline.generate_one", "make_unfit_rewriter"),
    ("nutrienv.bench.pipeline.generate_one", "parse_query_foods_payload"),
    ("nutrienv.bench.pipeline.generate_one", "search_fit_plate"),
    ("nutrienv.bench.pipeline.generate_one", "AMOUNT_PATHS"),
    ("nutrienv.bench.pipeline.generate_one", "KNIVES"),
    # nutrienv.bench.realize
    ("nutrienv.bench.realize", "compose_oracles"),
    ("nutrienv.bench.realize", "scored_oracles"),
    ("nutrienv.bench.realize", "realize_evaluate"),
    ("nutrienv.bench.realize", "bind_evaluate_reasons"),
    # nutrienv.bench.validator
    ("nutrienv.bench.validator", "validate_draft"),
    ("nutrienv.bench.validator", "semantic_key"),
    ("nutrienv.bench.validator", "fitting_plan"),
    # nutrienv.bench.pipeline.review_harness
    ("nutrienv.bench.pipeline.review_harness", "stage_a_code_gate"),
    # nutrienv.bench.quality_gates
    ("nutrienv.bench.quality_gates", "EVALUATE_TIERS"),
    # nutrienv.bench.pipeline.freezer
    ("nutrienv.bench.pipeline.freezer", "freeze_tasks"),
    ("nutrienv.bench.pipeline.freezer", "task_to_item"),
    # nutrienv.bench.pipeline.templates
    ("nutrienv.bench.pipeline.templates", "RECOMMEND_SHELLS"),
    ("nutrienv.bench.pipeline.templates", "UPDATE_SHELLS"),
    ("nutrienv.bench.pipeline.templates", "recommend_query"),
    ("nutrienv.bench.pipeline.templates", "update_query"),
    # nutrienv.bench.pipeline.roster
    ("nutrienv.bench.pipeline.roster", "ROSTER"),
    ("nutrienv.bench.pipeline.roster", "RosterPerson"),
    ("nutrienv.bench.pipeline.roster", "profile_for"),
    ("nutrienv.bench.pipeline.roster", "sample_roster_person"),
    # nutrienv.bench.pipeline.types
    ("nutrienv.bench.pipeline.types", "catalog_digest"),
    # nutrienv.world.daily_windows
    ("nutrienv.world.daily_windows", "plan_windows_for_meal"),
    ("nutrienv.world.daily_windows", "derive_profile_windows"),
    ("nutrienv.world.daily_windows", "meal_slot_and_remainder"),
    # nutrienv.world.types — MAX_ITEM_GRAMS (spec §18 and the pinned rev agree)
    # (typo; see module docstring)
    ("nutrienv.world.types", "ledger_totals"),
    ("nutrienv.world.types", "WorldState"),
    ("nutrienv.world.types", "Profile"),
    ("nutrienv.world.types", "LedgerRow"),
    ("nutrienv.world.types", "MAX_ITEM_GRAMS"),
    # nutrienv.world.catalog_store
    ("nutrienv.world.catalog_store", "load_catalog"),
    # nutrienv.harness (only ReActHarness and ScriptHarness are re-exported here)
    ("nutrienv.harness", "ReActHarness"),
    ("nutrienv.harness", "ScriptHarness"),
    # nutrienv.harness.tool_call / tools_schema — spec §18; listed in the v2
    # guard even if the lab modules have no __all__ (ADR-012)
    ("nutrienv.harness.tool_call", "run_episode_tool_call"),
    ("nutrienv.harness.tools_schema", "NUTRIENV_TOOLS"),
    ("nutrienv.harness.tools_schema", "TOOL_SYSTEM_PROMPT"),
    # nutrienv.harness.react — in that module's __all__, NOT re-exported from
    # nutrienv.harness (ticket 001 finding)
    ("nutrienv.harness.react", "react_manual"),
    ("nutrienv.harness.react", "context_messages"),
    # nutrienv.harness.runner
    ("nutrienv.harness.runner", "DEFAULT_MAX_STEPS"),
    ("nutrienv.harness.runner", "FAMILY_MAX_STEPS"),
    ("nutrienv.harness.runner", "FINISH_OPS"),
    # nutrienv.env
    ("nutrienv.env", "NutriEnv"),
]


def load_public_symbol(module: str, symbol: str) -> object:
    """Resolve one (module, symbol) pair of the public borrowed API.

    A module that stops importing, or a symbol that moves / renames /
    disappears, fails loudly with "public borrowed API broke: <module>.<symbol>"
    (ticket 016; spec §19.6). Shared with the pinned-signature test.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(
            f"public borrowed API broke: {module}.{symbol} "
            f"(module no longer imports: {exc})"
        )
    if not hasattr(mod, symbol):
        pytest.fail(f"public borrowed API broke: {module}.{symbol} (missing from {module})")
    return getattr(mod, symbol)


@pytest.mark.parametrize(("module", "symbol"), PUBLIC_BORROWED_API)
def test_public_borrowed_symbol_imports(module: str, symbol: str) -> None:
    """Every symbol of spec §18's public table resolves from its stated module."""
    load_public_symbol(module, symbol)


@pytest.mark.parametrize(("module", "symbol"), PUBLIC_BORROWED_API)
def test_public_borrowed_symbol_is_in_module_all(module: str, symbol: str) -> None:
    """Each borrowed symbol is genuinely public: in its module's ``__all__``.

    ADR-012 defines the guarded class as the symbols in nutri-env's `__all__`
    that v2 calls directly — a symbol quietly dropped from `__all__` while
    still importable would break that premise and must fail loudly too.
    """
    mod = importlib.import_module(module)
    exported = getattr(mod, "__all__", None)
    if exported is None:
        # ADR-012: FC harness symbols are in the v2 guard before the lab
        # exports them from __all__. Existence is still required.
        assert hasattr(mod, symbol), (
            f"public borrowed API broke: {module}.{symbol} (missing from {module})"
        )
        return
    assert symbol in set(exported or ()), (
        f"public borrowed API broke: {module}.{symbol} "
        f"(no longer in {module}.__all__)"
    )
