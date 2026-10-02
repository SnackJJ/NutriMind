"""Ticket 016 — compatibility guard (Seam 5), part 3: ADR-012 two-class rule.

This test lives in its own file rather than inside
`test_borrowed_api_signatures.py` because it is a test ABOUT the guard files
themselves: it introspects their pytest parametrization and scans their source
text. Keeping that meta-logic here leaves the signature file purely "one
pinned string per public symbol", and keeps the private-name checklist in
exactly one place — this file.

ADR-012 (amended 2026-09-08), two-class rule, encoded by the tests below:

- **Public class** — the borrowed symbols that are in nutri-env's `__all__`
  (spec §18's public table, `PUBLIC_BORROWED_API` in `test_imports.py`): a
  formal v2 dependency → import guard + pinned-signature guard
  (`test_imports.py`, `test_borrowed_api_signatures.py`) + behaviour tests
  elsewhere (Seam 1).
- **Private class** — the underscore-prefixed helpers of spec §18's private
  list (not in `__all__`): explicitly NOT frozen. No stability promise, no
  import, no signature pin, no reference anywhere in the guard files; covered
  indirectly by the Seam-1 end-to-end behaviour test. If a stable dependency
  on one becomes unavoidable, the correct move is to get it promoted to
  nutri-env's `__all__` upstream first — never to import the underscore name.

Imports here are the stdlib, pytest-free introspection, and sibling test
modules (nutrienv availability follows those modules' own skip convention), so
the file keeps the CI constraint: nothing beyond nutrienv + stdlib + siblings.
"""

from __future__ import annotations

import pathlib
import re

# Imported as module aliases (never `from ... import test_*`): binding a
# sibling module's test functions into this namespace would make pytest
# collect them a second time as tests of THIS file.
from tests.training.data_factory import (
    test_borrowed_api_signatures as signature_guard,
)
from tests.training.data_factory import test_imports as import_guard

_HERE = pathlib.Path(__file__).resolve().parent

# The guard files whose source text this test scans — this file included.
GUARD_FILES: tuple[pathlib.Path, ...] = (
    _HERE / "test_imports.py",
    _HERE / "test_borrowed_api_signatures.py",
    _HERE / "test_two_class_rule.py",
)

# Spec §18's private list (13 names), assembled from fragments so that the
# literal underscore-private nutri-env names appear NOWHERE in the guard
# files' source — not even in this checklist. That lets the scan below treat
# every match, anywhere in these files, as a violation with no exclusions
# (the naive alternative — literal names in a constant — would false-positive
# on its own checklist). The split-module names are matched bare with word
# boundaries: that catches attribute access, `from ... import <name>`, and
# getattr strings alike, while `\b` spares public look-alikes such as
# `task_to_item` or `sub_oracles`.
_PRIVATE_NAME_PARTS: tuple[tuple[str, ...], ...] = (
    ("update", "from", "template"),
    ("bind", "log", "foods"),
    ("log", "then", "recommend"),
    ("update", "then", "recommend"),
    ("composite", "speech", "spans"),
    ("recommend", "from", "template"),
    ("evaluate", "from", "bound"),
    ("anchored", "bind", "grams"),
    ("parse", "action"),
    ("SYSTEM", "V2"),
    ("item",),
    ("s0",),
    ("oracle",),
)
PRIVATE_NUTRIENV_NAMES: tuple[str, ...] = tuple(
    "_" + "_".join(parts) for parts in _PRIVATE_NAME_PARTS
)

_WORD_BOUNDARY_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(rf"\b{re.escape(name)}\b") for name in PRIVATE_NUTRIENV_NAMES
}


def _private_name_violations(text: str) -> list[str]:
    """Which private-list names occur in `text` as whole identifiers?"""
    return [
        name for name, pattern in _WORD_BOUNDARY_PATTERNS.items() if pattern.search(text)
    ]


def _parametrize_cases(testfunc: object) -> list[tuple[str, ...]]:
    """The actual parametrize cases pytest recorded on a test function.

    Reads the `parametrize` mark back off the decorated function so the
    assertions below check what CI really runs — not merely what a constant
    claims it runs.
    """
    marks = [
        m for m in getattr(testfunc, "pytestmark", ()) if getattr(m, "name", None) == "parametrize"
    ]
    assert len(marks) == 1, (
        f"expected exactly one parametrize mark on {testfunc.__name__}, "  # type: ignore[attr-defined]
        f"found {len(marks)}"
    )
    argvalues = marks[0].args[1]
    cases = []
    for case in argvalues:
        values = getattr(case, "values", None)  # unwrap pytest.param(...) if used
        cases.append(tuple(values if values is not None else case))
    return cases


def test_public_class_is_signature_guarded() -> None:
    """Public → import guard + signature guard, over exactly the same table.

    The pinned baseline and every guard test's actual pytest parametrization
    must each cover EXACTLY `PUBLIC_BORROWED_API` (set equality): a public
    symbol missing from any of them is an unguarded dependency; an extra
    entry freezes something the spec's public table never promised.
    """
    public = set(import_guard.PUBLIC_BORROWED_API)
    assert len(import_guard.PUBLIC_BORROWED_API) == len(public), (
        "PUBLIC_BORROWED_API contains duplicate (module, symbol) entries"
    )

    assert set(signature_guard.EXPECTED_SIGNATURES) == public, (
        "EXPECTED_SIGNATURES must pin exactly PUBLIC_BORROWED_API — "
        f"missing: {sorted(public - set(signature_guard.EXPECTED_SIGNATURES))}, "
        f"unpinned-in-spec: {sorted(set(signature_guard.EXPECTED_SIGNATURES) - public)}"
    )

    for func in (
        import_guard.test_public_borrowed_symbol_imports,
        import_guard.test_public_borrowed_symbol_is_in_module_all,
        signature_guard.test_public_borrowed_signature_is_pinned,
    ):
        cases = set(_parametrize_cases(func))
        assert cases == public, (
            f"{func.__name__} must parametrize exactly PUBLIC_BORROWED_API — "
            f"missing: {sorted(public - cases)}, extra: {sorted(cases - public)}"
        )


def test_private_class_is_not_frozen() -> None:
    """Private → not frozen: no private-list name may appear in any guard file.

    Reading the guard files' own source text (all three, including this one)
    and asserting none of spec §18's private names occurs guarantees the guard
    freezes nothing private: no import, no signature pin, not even a comment
    reference. Private helpers are covered indirectly by the Seam-1
    end-to-end behaviour test instead.
    """
    for path in GUARD_FILES:
        violations = _private_name_violations(path.read_text(encoding="utf-8"))
        assert not violations, (
            f"{path.name} references private nutri-env symbol(s) {violations}: "
            "private helpers are not frozen by this guard (ADR-012 two-class "
            "rule); if a stable dependency becomes unavoidable, promote the "
            "symbol to nutri-env's __all__ upstream instead"
        )


def test_scanner_detects_names_and_spares_public_lookalikes() -> None:
    """Sanity for the scanner itself (guards the guard).

    Every private name is caught in a synthetic violation, and the public
    look-alikes the guard files legitimately contain — longer identifiers
    ending in the same characters (`task_to_item`, `sub_oracles`,
    `compose_oracles`, `scored_oracles`, `*oracles`) and the bare public
    `s0` — produce no match.
    """
    assert len(PRIVATE_NUTRIENV_NAMES) == 13
    assert all(name.startswith("_") for name in PRIVATE_NUTRIENV_NAMES)

    dirty = "\n".join(f"result = {name}(task)" for name in PRIVATE_NUTRIENV_NAMES)
    assert set(_private_name_violations(dirty)) == set(PRIVATE_NUTRIENV_NAMES)

    lookalikes = (
        "task_to_item\n"
        "sub_oracles: 'tuple[Oracle, ...] | None'\n"
        "compose_oracles, scored_oracles\n"
        "*oracles: 'Oracle'\n"
        "s0: 'WorldState'\n"
    )
    assert _private_name_violations(lookalikes) == []
