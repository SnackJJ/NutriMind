"""Ticket 016 — compatibility guard (Seam 5), part 2: pinned signatures.

Baseline captured at nutrienv rev 47367d9c569d0a46cbd1c97d5f08afb3a7d573ac
(the pin in `configs/data_factory.yaml`; the installed-rev == pin assertion
lives in `test_nutrienv_smoke.py`, which CI runs alongside this file). An
upstream change that breaks any borrowed signature must fail CI loudly.

Parametrized over the SAME `PUBLIC_BORROWED_API` table as `test_imports.py`
(the table is imported, never duplicated); `test_two_class_rule.py` asserts
the parametrization and this baseline cover exactly that table.

Per-symbol pinned string: for callables it is ``str(inspect.signature(obj))``.
Module-level constants have no signature, so their pin is
``<constant:<type name>>`` — enough to catch a constant becoming a callable or
changing kind, without freezing its value (value wiring stays in the smoke
test, e.g. `FAMILY_MAX_STEPS["composite"]`).

ADR-012 (amended) two-class rule: ONLY the public borrowed API is
signature-guarded here. The underscore-prefixed private helpers of spec §18's
private list are explicitly NOT frozen — no signature assertion for them, no
reference to them anywhere in the guard files (`test_two_class_rule.py`
asserts that). They are covered indirectly by the Seam-1 end-to-end behaviour
test.

No behaviour tests live in this file (Seam 5 is split from behaviour):
imports + signatures only.

Regenerating the baseline after a deliberate rev bump — print and diff:

    .venv/bin/python - <<'EOF'
    import importlib, inspect
    from tests.training.data_factory.test_imports import PUBLIC_BORROWED_API
    for module, symbol in PUBLIC_BORROWED_API:
        obj = getattr(importlib.import_module(module), symbol)
        pin = (str(inspect.signature(obj)) if callable(obj)
               else f"<constant:{type(obj).__name__}>")
        print(f'    ("{module}", "{symbol}"): "{pin}",')
    EOF

CI constraint: this file runs in a fresh venv holding ONLY pytest + nutrienv —
it imports nothing beyond nutrienv, the stdlib, pytest, and sibling test
modules.
"""

from __future__ import annotations

import inspect

import pytest

from tests.training.data_factory.test_imports import (
    PUBLIC_BORROWED_API,
    load_public_symbol,
)

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

# The rev the baseline below was captured from (configs/data_factory.yaml pin).
NUTRIENV_BASELINE_REV = "47367d9c569d0a46cbd1c97d5f08afb3a7d573ac"

# One pinned string per (module, symbol) — keys MUST equal PUBLIC_BORROWED_API
# exactly (asserted by test_two_class_rule.py). Order mirrors that table.
EXPECTED_SIGNATURES: dict[tuple[str, str], str] = {
    ("nutrienv.bench", "Oracle"): "(profile: 'Profile | None' = None, last_plan: 'list | None' = None, ledger_tail: 'list | None' = None, ledger: 'tuple[LedgerRow, ...] | None' = None, plan_must_be_safe: 'bool' = False, plan_must_fit_windows: 'bool' = False, allow_empty_plan: 'bool' = False, plan_windows: 'dict[str, tuple[float, float]] | None' = None, last_verdict: 'str | None' = None, last_reasons: 'tuple[str, ...]' = (), update_band: 'str | None' = None, evaluated_plan: 'list | None' = None, bound_labels: 'tuple[str, ...]' = (), sub_oracles: 'tuple[Oracle, ...] | None' = None, allowed_food_ids: 'frozenset[str] | None' = None) -> None",
    ("nutrienv.bench", "Task"): "(id: 'str', family: 'str', query: 'str', s0: 'WorldState', oracle: 'Oracle', situations: 'tuple[str, ...]' = (), persona: 'str' = 'everyday', tier: 'str' = '') -> None",
    ("nutrienv.bench", "Scorer"): "()",
    ("nutrienv.bench", "check_achievable"): "(tasks: 'Sequence[Task]') -> 'AchievabilityReport'",
    ("nutrienv.bench", "load_split"): "(path: 'Path | str | None' = None, *, catalog=None) -> 'list[Task]'",
    ("nutrienv.bench", "load_exam"): "(path: 'Path | str | None' = None) -> 'list[Task]'",
    ("nutrienv.bench", "EXAM_SPLIT_PATH"): "<constant:PosixPath>",
    ("nutrienv.bench.pipeline.generate_one", "generate_one"): "(*, catalog: 'Mapping', expander: 'Callable[..., object] | None' = None, family: 'str' = 'log', seed: 'int' = 0, person: 'RosterPerson | None' = None, amount_path: 'str | None' = None, occasion: 'str' = 'lunch', pool_size: 'int' = 12, knife: 'str | None' = None, rewriter: 'Callable[..., object] | None' = None, scene: 'str' = 'empty', prior_ledger: 'Sequence[LedgerRow] | None' = None, prior_logs: 'Sequence[Task] | None' = None, last_meal: 'bool' = False, shell: 'str | None' = None, slots: 'Mapping[str, str] | None' = None, steps: 'Sequence[str] | None' = None, tier: 'str' = '', gram_anchor: 'GramAnchor | None' = None, items: 'Sequence[Mapping[str, object]] | None' = None, enable_semantic_vote: 'bool' = False) -> 'GenerateOneResult'",
    ("nutrienv.bench.pipeline.generate_one", "make_log_expander"): "(*, complete: 'Callable[[str, Sequence[Mapping[str, str]]], str]', parse_retries: 'int' = 1, model_id: 'str | None' = None) -> 'LogExpander'",
    ("nutrienv.bench.pipeline.generate_one", "make_unfit_rewriter"): "(*, complete: 'Callable[[str, Sequence[Mapping[str, str]]], str]', catalog: 'Mapping', model_id: 'str' = 'qwen3.8-max', parse_retries: 'int' = 1) -> 'UnfitRewriter'",
    ("nutrienv.bench.pipeline.generate_one", "parse_query_foods_payload"): "(payload: 'object') -> 'dict[str, object] | None'",
    ("nutrienv.bench.pipeline.generate_one", "search_fit_plate"): "(pool: 'FoodPool', *, profile: 'Profile', catalog: 'Mapping', occasion: 'str', last_meal: 'bool' = False, max_foods: 'int' = 3) -> 'list[dict[str, object]] | None'",
    ("nutrienv.bench.pipeline.generate_one", "AMOUNT_PATHS"): "<constant:tuple>",
    ("nutrienv.bench.pipeline.generate_one", "KNIVES"): "<constant:tuple>",
    ("nutrienv.bench.realize", "compose_oracles"): "(*oracles: 'Oracle') -> 'Oracle'",
    ("nutrienv.bench.realize", "scored_oracles"): "(oracle: 'Oracle') -> 'tuple[Oracle, ...]'",
    ("nutrienv.bench.realize", "realize_evaluate"): "(*, task_id: 'str', query: 'str', items: 'list', s0: 'WorldState', occasion: 'str', last_meal: 'bool' = False, tier: 'str' = '') -> 'Task'",
    ("nutrienv.bench.realize", "bind_evaluate_reasons"): "(items: 'list', windows: 'dict[str, tuple[float, float]]', catalog: 'Mapping', allergies: 'tuple[str, ...]') -> 'tuple[str, ...]'",
    ("nutrienv.bench.validator", "validate_draft"): "(task: 'Task') -> 'list[str]'",
    ("nutrienv.bench.validator", "semantic_key"): "(task: 'Task') -> 'tuple'",
    ("nutrienv.bench.validator", "fitting_plan"): "(catalog, windows: 'dict', allergies, allowed_food_ids: 'frozenset[str] | None' = None) -> 'list[dict] | None'",
    ("nutrienv.bench.pipeline.review_harness", "stage_a_code_gate"): "(task: 'Task') -> 'list[str]'",
    ("nutrienv.bench.quality_gates", "EVALUATE_TIERS"): "<constant:tuple>",
    ("nutrienv.bench.pipeline.freezer", "freeze_tasks"): "(tasks: 'Sequence[Task]', *, catalog, catalog_field: 'str' = 'data/fdc/archive/catalog-v1.sqlite', catalog_sha: 'str | None' = None, output_path: 'Path | str | None' = None, extra: 'Mapping[str, object] | None' = None, overwrite: 'bool' = False) -> 'tuple[dict, Path]'",
    ("nutrienv.bench.pipeline.freezer", "task_to_item"): "(task: 'Task') -> 'dict'",
    ("nutrienv.bench.pipeline.templates", "RECOMMEND_SHELLS"): "<constant:dict>",
    ("nutrienv.bench.pipeline.templates", "UPDATE_SHELLS"): "<constant:dict>",
    ("nutrienv.bench.pipeline.templates", "recommend_query"): "(shell: 'str', slots: 'dict[str, str]') -> 'str | None'",
    ("nutrienv.bench.pipeline.templates", "update_query"): "(shell: 'str', slots: 'dict[str, str]') -> 'str | None'",
    ("nutrienv.bench.pipeline.roster", "ROSTER"): "<constant:tuple>",
    ("nutrienv.bench.pipeline.roster", "RosterPerson"): "(user_id: 'str', sex: 'str', age_y: 'int', height_cm: 'float', weight_kg: 'float', activity: 'str', phase: 'str' = 'maintain', allergies: 'tuple[str, ...]' = (), persona: 'str' = 'everyday', diet_style: 'str' = 'standard') -> None",
    ("nutrienv.bench.pipeline.roster", "profile_for"): "(person: 'RosterPerson', *, user_id: 'str | None' = None) -> 'Profile'",
    ("nutrienv.bench.pipeline.roster", "sample_roster_person"): "(seed: 'int') -> 'RosterPerson'",
    ("nutrienv.bench.pipeline.types", "catalog_digest"): "(catalog) -> 'str'",
    ("nutrienv.world.daily_windows", "plan_windows_for_meal"): "(daily: 'dict[str, tuple[float, float]]', eaten: 'dict[str, float]', occasion: 'str', *, last_meal: 'bool' = False) -> 'dict[str, tuple[float, float]] | None'",
    ("nutrienv.world.daily_windows", "derive_profile_windows"): "(profile: 'Profile') -> 'dict[str, tuple[float, float]] | None'",
    ("nutrienv.world.daily_windows", "meal_slot_and_remainder"): "(daily: 'dict[str, tuple[float, float]]', eaten: 'dict[str, float]', occasion: 'str') -> 'tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]'",
    ("nutrienv.world.types", "ledger_totals"): "(rows: 'list[LedgerRow]', catalog: 'dict') -> 'dict[str, float]'",
    ("nutrienv.world.types", "WorldState"): "(profile: 'Profile', ledger: 'list[LedgerRow]' = <factory>, catalog: 'Mapping[str, dict]' = <factory>, last_plan: 'list' = <factory>, last_verdict: 'str | None' = None, last_reasons: 'tuple[str, ...]' = (), allowed_food_ids: 'frozenset[str] | None' = None) -> None",
    ("nutrienv.world.types", "Profile"): "(user_id: 'str', allergies: 'tuple[str, ...]' = (), medications: 'tuple[str, ...]' = (), windows: 'dict[str, tuple[float, float]]' = <factory>, plan_preset: 'dict' = <factory>, version: 'int' = 1, sex: 'str | None' = None, age_y: 'int | None' = None, height_cm: 'float | None' = None, weight_kg: 'float | None' = None, activity: 'str | None' = None, phase: 'str' = 'maintain') -> None",
    ("nutrienv.world.types", "LedgerRow"): "(food_id: 'str', grams: 'float', eaten_at: 'str') -> None",
    ("nutrienv.world.types", "MAX_ITEM_GRAMS"): "<constant:float>",
    ("nutrienv.world.catalog_store", "load_catalog"): "(path: 'Path | str | None' = None, *, demo: 'bool' = False) -> 'FoodCatalog'",
    ("nutrienv.harness", "ReActHarness"): "(*, api_key: 'str | None' = None, base_url: 'str | None' = None, model: 'str' = 'deepseek-chat', timeout: 'float' = 60.0, leak_oracle: 'bool' = False, max_steps: 'int' = 12, extra_body: 'dict | None' = None, version: 'str' = 'v0', context_limit: 'int | None' = None, temperature: 'float' = 0.0) -> 'None'",
    ("nutrienv.harness", "ScriptHarness"): "()",
    ("nutrienv.harness.tool_call", "run_episode_tool_call"): "(task, harness_spec: 'dict[str, Any]', catalog: 'Any', step_telemetry_cls: 'Any', task_telemetry_cls: 'Any') -> 'Any'",
    ("nutrienv.harness.tools_schema", "NUTRIENV_TOOLS"): "<constant:list>",
    ("nutrienv.harness.tools_schema", "TOOL_SYSTEM_PROMPT"): "<constant:str>",
    ("nutrienv.harness.react", "react_manual"): "(version: 'str') -> 'str'",
    ("nutrienv.harness.react", "context_messages"): "(messages: 'list[dict]', *, limit: 'int | None' = None) -> 'list[dict]'",
    ("nutrienv.harness.runner", "DEFAULT_MAX_STEPS"): "<constant:int>",
    ("nutrienv.harness.runner", "FAMILY_MAX_STEPS"): "<constant:dict>",
    ("nutrienv.harness.runner", "FINISH_OPS"): "<constant:frozenset>",
    ("nutrienv.env", "NutriEnv"): "(*, default_eaten_at: 'str' = 'now') -> 'None'",
}


def pinned_descriptor(obj: object) -> str:
    """The pinned string for one symbol: signature for callables, type marker
    for constants (see module docstring)."""
    if callable(obj):
        return str(inspect.signature(obj))
    return f"<constant:{type(obj).__name__}>"


@pytest.mark.parametrize(("module", "symbol"), PUBLIC_BORROWED_API)
def test_public_borrowed_signature_is_pinned(module: str, symbol: str) -> None:
    """`inspect.signature` of every public borrowed symbol equals its pin.

    A mismatch means upstream (at the pinned rev) changed a public borrowed
    signature — CI must fail loudly, and the baseline must be regenerated and
    reviewed as part of the deliberate rev bump.
    """
    try:
        expected = EXPECTED_SIGNATURES[(module, symbol)]
    except KeyError:
        pytest.fail(
            f"no pinned signature for {module}.{symbol} — EXPECTED_SIGNATURES "
            f"must cover PUBLIC_BORROWED_API exactly (see module docstring "
            f"for the regeneration snippet)"
        )
    actual = pinned_descriptor(load_public_symbol(module, symbol))
    assert actual == expected, (
        f"public borrowed API broke: signature of {module}.{symbol} changed "
        f"(baseline rev {NUTRIENV_BASELINE_REV})\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )
