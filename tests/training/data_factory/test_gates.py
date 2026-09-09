"""Ticket 005 — gates.run pure validation seam (Seam 2, spec §11).

Every gate construction below was verified against the pinned nutrienv rev:
the exam corpus facts (63 normalized queries / 62 semantic keys — the frozen
exam itself contains one intra-duplicate pair — / update-slot values), and the
six single-gate trip constructions. All offline; the expander is synthetic.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import load_exam  # noqa: E402
from nutrienv.bench.validator import semantic_key  # noqa: E402

from src.training.data_factory import gates  # noqa: E402
from src.training.data_factory.concepts import GateResult  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402

# verified at rev 203d807: exam update items used exactly these slot values
EXAM_SLOT_VALUES = {"egg", "milk", "peanut", "shellfish", "tree_nut"}


@pytest.fixture(scope="module")
def catalog():
    return fx.gold_catalog()


@pytest.fixture(scope="module")
def person():
    return fx.first_person(require_no_allergies=True)


@pytest.fixture(scope="module")
def exam_ctx():
    return gates.GateContext.from_exam(load_exam())


@pytest.fixture(scope="module")
def log_task(catalog, person):
    return fx.make_log_task(catalog, person, seed=30)


# --------------------------------------------------------------------------- #
# GateContext.from_exam
# --------------------------------------------------------------------------- #


def test_from_exam_precomputes_corpus_facts(exam_ctx):
    assert len(exam_ctx.normalized_exam_queries) == 63
    # the frozen exam itself contains one intra-duplicate semantic_key pair
    # (adr20-rec-5019 / adr29-conv-03 — two recommend tasks, same persona and
    # windows); 63 tasks → 62 distinct keys. That is the exam's business.
    assert len(exam_ctx.exam_semantic_keys) == 62
    assert set(exam_ctx.exam_update_slot_values) == EXAM_SLOT_VALUES


def test_from_exam_accepts_handpicked_list(catalog, person, log_task):
    """Tests build a GateContext from a hand-picked task list — no file I/O."""
    update = fx.make_update_task(
        catalog, person, seed=41, shell="upd-add-allergy-short", slots={"allergen": "soy"}
    )
    ctx = gates.GateContext.from_exam([log_task, update])
    assert len(ctx.normalized_exam_queries) == 2
    assert len(ctx.exam_semantic_keys) == 2
    assert ctx.exam_update_slot_values == frozenset({"soy"})


def test_run_is_pure(catalog, person, log_task, exam_ctx):
    """Same (task, ctx) → same GateResult; independent context builds agree."""
    again = gates.GateContext.from_exam(load_exam())
    results = [gates.run(log_task, ctx) for ctx in (exam_ctx, exam_ctx, again)]
    assert results[0] == results[1] == results[2]
    assert isinstance(results[0], GateResult)


# --------------------------------------------------------------------------- #
# the six gates, one trip each
# --------------------------------------------------------------------------- #


def test_clean_log_task_kept(log_task, exam_ctx):
    result = gates.run(log_task, exam_ctx)
    assert result.keep is True
    assert result.failure_code is None


def test_verbatim_query_collision(log_task, exam_ctx):
    exam_query = next(t.query for t in load_exam() if t.id == "adr20-upd-5026")
    assert exam_query  # e.g. "Add peanut to my allergies."
    candidate = dataclasses.replace(log_task, query=exam_query)
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.verbatim_query_collision"

    # normalization is what collides: case / whitespace / trailing punctuation
    mutated = dataclasses.replace(
        log_task, query=f"  {exam_query.upper()}  !!! "
    )
    again = gates.run(mutated, exam_ctx)
    assert again.failure_code == "gate.verbatim_query_collision"


def test_near_duplicate_is_kept(log_task, exam_ctx):
    """Near-duplicates that are not verbatim are deliberately kept (spec §11)."""
    exam_query = next(t.query for t in load_exam() if t.id == "adr20-upd-5026")
    candidate = dataclasses.replace(
        log_task, query=f"Hey there! {exam_query} Could you help?"
    )
    result = gates.run(candidate, exam_ctx)
    assert result.keep is True, result


def test_semantic_key_collision(catalog, person, exam_ctx):
    """update add-egg has the same structural semantic_key as the frozen exam
    item adr20-upd-5027 (the update branch keys on the profile diff, not the
    query) — rewording the query must NOT rescue it."""
    candidate = fx.make_update_task(
        catalog, person, seed=42, shell="upd-add-allergy-short", slots={"allergen": "egg"}
    )
    candidate = dataclasses.replace(candidate, query=candidate.query + " Thanks so much!")
    assert normalize_query_differs(candidate.query, exam_ctx)
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.semantic_key_collision"
    assert semantic_key(candidate) in exam_ctx.exam_semantic_keys


def normalize_query_differs(query, ctx) -> bool:
    return gates.normalize_query(query) not in ctx.normalized_exam_queries


def test_slot_value_overlaps_exam(catalog, person, exam_ctx):
    """update add-milk: semantic_key is unique ('milk' was only used by exam
    COMPOSITE update legs, whose keys take the composite fallback branch), but
    the slot value itself is an exam update value → gate 3."""
    candidate = fx.make_update_task(
        catalog, person, seed=43, shell="upd-add-allergy-short", slots={"allergen": "milk"}
    )
    candidate = dataclasses.replace(candidate, query=candidate.query + " Please!")
    assert semantic_key(candidate) not in exam_ctx.exam_semantic_keys
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.slot_value_overlaps_exam"
    assert "milk" in result.reason_detail


def test_stage_a_code_gate(log_task, exam_ctx):
    """Off-portion-table ledger grams trip stage_a (gate 4 fires before the
    achievability gate even though the mutation is also unreachable)."""
    rows = tuple(
        dataclasses.replace(row, grams=row.grams + 0.37)
        for row in log_task.oracle.ledger_tail
    )
    candidate = dataclasses.replace(
        log_task, oracle=dataclasses.replace(log_task.oracle, ledger_tail=rows)
    )
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.stage_a"
    assert "grams_off_table" in result.reason_detail


def test_draft_invalid(log_task, exam_ctx):
    candidate = dataclasses.replace(
        log_task, query="Please log food_id 12345 for lunch."
    )
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.draft_invalid"
    assert "food_id" in result.reason_detail


def test_unachievable_routes_indeterminate(catalog, exam_ctx):
    """A composite whose update leg carries STALE kcal windows (declared in the
    query, so only the allow-listed draft FP remains) passes every static gate
    but cannot replay — an authoring bug → indeterminate, not a plain drop."""
    task, reason = fx.assemble_three_leg(catalog, seed=101, allergen="fish")
    assert task is not None, f"3-leg assembly failed: {reason}"

    upd_sub, log_sub, rec_sub = task.oracle.sub_oracles
    stale = dict(upd_sub.profile.windows)
    stale["kcal"] = (
        upd_sub.profile.windows["kcal"][0] - 555.0,
        upd_sub.profile.windows["kcal"][1] - 555.0,
    )
    bad_update = dataclasses.replace(
        upd_sub, profile=dataclasses.replace(upd_sub.profile, windows=stale)
    )
    bad_log = dataclasses.replace(
        log_sub, profile=copy.deepcopy(bad_update.profile)
    )
    candidate = dataclasses.replace(
        task,
        id="3leg-stale-kcal--001",
        query=task.query + " My doctor told me to lower my calorie target by 555.",
        oracle=dataclasses.replace(
            task.oracle, sub_oracles=[bad_update, bad_log, rec_sub]
        ),
    )
    result = gates.run(candidate, exam_ctx)
    assert result.keep is False
    assert result.failure_code == "gate.unachievable"

    record = gates.rejects_record(result, candidate)
    assert record["status"] == "indeterminate"  # routed to indeterminate, not dropped
    assert record["failure_codes"] == ["gate.unachievable"]


def test_three_leg_passes_with_draft_allowlist(catalog, exam_ctx):
    """The 3-leg composite trips exactly the allow-listed validate_draft
    false-positive and must otherwise pass every gate (spec §22.9)."""
    task, reason = fx.assemble_three_leg(catalog, seed=101, allergen="fish")
    assert task is not None, f"3-leg assembly failed: {reason}"
    result = gates.run(task, exam_ctx)
    assert result.keep is True, result


# --------------------------------------------------------------------------- #
# ordering + reject record shape
# --------------------------------------------------------------------------- #


def test_first_failure_wins(log_task, exam_ctx):
    """A task failing check 1 (verbatim) and check 4 (stage_a) reports check 1."""
    exam_query = next(t.query for t in load_exam() if t.id == "adr20-upd-5026")
    rows = tuple(
        dataclasses.replace(row, grams=row.grams + 0.37)
        for row in log_task.oracle.ledger_tail
    )
    candidate = dataclasses.replace(
        log_task,
        query=exam_query,
        oracle=dataclasses.replace(log_task.oracle, ledger_tail=rows),
    )
    result = gates.run(candidate, exam_ctx)
    assert result.failure_code == "gate.verbatim_query_collision"


def test_rejects_record_shapes(log_task, exam_ctx):
    dropped = gates.run(
        dataclasses.replace(log_task, query="Please log food_id 12345 for lunch."),
        exam_ctx,
    )
    record = gates.rejects_record(dropped, log_task, intent={"family": "log", "seed": 30})
    assert record["schema_version"] == "nutrimind-v2-reject/1"
    assert record["stage"] == "gate"
    assert record["status"] == "dropped"
    assert record["failure_codes"] == ["gate.draft_invalid"]
    assert record["intent"] == {"family": "log", "seed": 30}

    with pytest.raises(ValueError):
        gates.rejects_record(gates.run(log_task, exam_ctx), log_task)
