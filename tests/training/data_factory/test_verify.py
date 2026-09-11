"""Ticket 007 — tri-state verifier (Seam 3, spec §11 / §12 / §19.2 / §19.3).

Offline: episodes are driven through the real ``NutriEnv`` by scripted actions
(the scripted-teacher pattern ticket 009 will formalize), packages come from
the ticket-006 materializer, and every verdict is asserted on all three axes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("nutrienv", reason="run scripts/setup_nutrienv.sh")

from nutrienv.bench import Scorer, load_exam  # noqa: E402
from nutrienv.bench.pipeline.generate_one import generate_one  # noqa: E402
from nutrienv.bench.validator import fitting_plan  # noqa: E402
from nutrienv.env import NutriEnv  # noqa: E402
from nutrienv.harness.runner import FINISH_OPS  # noqa: E402
from nutrienv.world.catalog_store import GOLD_CATALOG_PATH, load_catalog  # noqa: E402

from src.training.data_factory import gates, materialize as mz, verify as vf  # noqa: E402
from src.training.data_factory.concepts import (  # noqa: E402
    EpisodeResult,
    TurnMeta,
)
from src.training.data_factory.materialize import RunContext  # noqa: E402
from src.training.data_factory.roster_train import TRAIN_ROSTER  # noqa: E402

from tests.training.data_factory import _fixtures as fx  # noqa: E402

NUTRIENV_REV = "203d807b19953a86b5486303ba6f7dd3b9cf7bb6"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(GOLD_CATALOG_PATH)


def make_ctx(catalog, **overrides) -> RunContext:
    fields = dict(
        catalog=catalog,
        catalog_sha=mz.catalog_digest(catalog),
        nutrienv_rev=NUTRIENV_REV,
        nutrimind_rev="deadbeef" * 5,
        config_sha="0" * 64,
        built_at="2026-09-09T12:00:00+00:00",
    )
    fields.update(overrides)
    return RunContext(**fields)


def run_episode(task, actions, *, error=None, reached_finish=None, raw_overrides=None):
    """Scripted rollout: drive the real env, record TurnMeta per turn."""
    env = NutriEnv()
    env.reset(task.s0)
    turns = []
    for index, action in enumerate(actions):
        env.step(action)
        raw = json.dumps(action)
        if raw_overrides and index in raw_overrides:
            raw = raw_overrides[index]
        turns.append(TurnMeta(raw_action_text=raw, executed_op=action))
    if reached_finish is None:
        reached_finish = bool(actions) and actions[-1].get("op") in FINISH_OPS
    return EpisodeResult(
        end_state=env.state(),
        turns=turns,
        reached_finish=reached_finish,
        error=error,
        task=task,
    )


def pkg_for(task, catalog, *, seed=1, steps=()):
    return mz.materialize(task, make_ctx(catalog, seed=seed, steps=steps))


@pytest.fixture(scope="module")
def log_task(catalog):
    person = next(p for p in TRAIN_ROSTER if p.persona == "everyday" and not p.allergies)
    return fx.make_log_task(catalog, person, seed=30)


@pytest.fixture(scope="module")
def rec_task(catalog):
    person = next(p for p in TRAIN_ROSTER if p.persona == "gym" and not p.allergies)
    result = generate_one(
        catalog=catalog, family="recommend", person=person, seed=12,
        occasion="dinner", shell="rec-dinner",
    )
    assert result.accepted is not None, result.rejected
    return result.accepted


@pytest.fixture(scope="module")
def update_task(catalog):
    person = next(p for p in TRAIN_ROSTER if p.persona == "everyday" and not p.allergies)
    result = generate_one(
        catalog=catalog, family="update", person=person, seed=41,
        shell="upd-add-allergy-short", slots={"allergen": "soy"},
    )
    assert result.accepted is not None, result.rejected
    return result.accepted


def _log_actions(task):
    return [
        {"op": "log_meal", "food_id": row.food_id, "grams": row.grams,
         "eaten_at": row.eaten_at}
        for row in task.oracle.ledger_tail
    ] + [{"op": "done"}]


def _update_actions(task):
    patch = {}
    expected = task.oracle.profile
    current = task.s0.profile
    for field in ("allergies", "windows", "weight_kg", "phase"):
        if getattr(expected, field, None) != getattr(current, field, None):
            value = getattr(expected, field)
            patch[field] = (
                list(value) if field == "allergies" else dict(value)
                if field == "windows" else value
            )
    return [{"op": "update_profile", "patch": patch}, {"op": "done"}]


# --------------------------------------------------------------------------- #
# derive_execution (pure helper)
# --------------------------------------------------------------------------- #


def test_derive_execution_matrix(log_task):
    ok = run_episode(log_task, _log_actions(log_task))
    assert vf.derive_execution(ok) == "ok"

    no_finish = run_episode(log_task, _log_actions(log_task)[:-1], reached_finish=False)
    assert vf.derive_execution(no_finish) == "no_finish"

    cap = run_episode(log_task, _log_actions(log_task)[:-1] + [{"op": "get_profile"}])
    assert vf.derive_execution(cap) == "no_finish"

    errored = run_episode(log_task, _log_actions(log_task), error="api timeout")
    assert vf.derive_execution(errored) == "error"

    # a recorded error outranks everything
    errored_finish = run_episode(log_task, _log_actions(log_task), error="api timeout")
    assert vf.derive_execution(errored_finish) == "error"


def test_parse_action_text_vocabulary():
    ok, status = vf.parse_action_text('{"op": "log_meal", "food_id": "1", "grams": 100}')
    assert status == "ok" and ok["op"] == "log_meal"

    # fenced JSON, prose around it
    ok2, status2 = vf.parse_action_text(
        'Sure! ```json\n{"op": "done"}\n``` hope that helps'
    )
    assert status2 == "ok" and ok2 == {"op": "done"}

    assert vf.parse_action_text("no json here at all") == (None, "no_json")
    assert vf.parse_action_text('{"food_id": "1"}') == (None, "no_op")
    assert vf.parse_action_text('{"op": "reboot_server"}') == (None, "illegal_op")
    assert vf.parse_action_text("") == (None, "empty")

    # the submit_plan reasons normalization mirrors the harness's executed form
    plan, status = vf.parse_action_text(
        '{"op": "submit_plan", "verdict": "accept", "reasons": ["a"], "items": []}'
    )
    assert status == "ok" and "reasons" not in plan


# --------------------------------------------------------------------------- #
# pass / fail on real episodes
# --------------------------------------------------------------------------- #


def test_legal_log_episode_passes(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, _log_actions(log_task)))
    assert result.status == "pass"
    assert (result.execution, result.oracle_exec, result.scorer) == ("ok", "ok", "pass")
    assert result.reward == 1.0
    assert result.failure_codes == []
    assert result.diagnostic_scores is None
    assert result.oracle_version == "nutrienv-203d807"
    assert result.rubric_version == result.reward_version == "v2-r1"


def test_verify_reads_end_state_not_tool_calls(catalog, log_task):
    """Ticket 024: verify is invariant to tool_calls; it scores end_state."""
    pkg = pkg_for(log_task, catalog, seed=30)
    episode = run_episode(log_task, _log_actions(log_task))
    baseline = vf.verify(pkg, episode)
    episode.turns[0].tool_calls = [{"id": "bogus", "type": "function"}]
    episode.turns[0].tool_call_id = "bogus"
    again = vf.verify(pkg, episode)
    assert again.status == baseline.status
    assert again.scorer == baseline.scorer
    assert again.reward == baseline.reward


def test_ledger_gram_tolerance_just_inside_passes(catalog, log_task):
    """±15 % gram tolerance (ADR-0023): 1.14x stays a multiset match → pass."""
    actions = _log_actions(log_task)
    for action in actions:
        if action.get("op") == "log_meal":
            action["grams"] = round(action["grams"] * 1.14, 2)
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, actions))
    assert result.status == "pass", result.evidence


def test_ledger_gram_beyond_tolerance_fails(catalog, log_task):
    actions = _log_actions(log_task)
    for action in actions:
        if action.get("op") == "log_meal":
            action["grams"] = round(action["grams"] * 1.2, 2)
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, actions))
    assert result.status == "fail"
    assert result.execution == "ok"  # legal episode, real model failure
    assert result.failure_codes == ["task_fail", "log_miss"]
    assert result.reward == 0.0
    # evidence carries the concrete rows that missed
    detail = result.evidence[-1]
    assert "expected_tail" in detail and "ledger_tail" in detail


def test_legal_update_episode_passes(catalog, update_task):
    pkg = pkg_for(update_task, catalog, seed=41)
    result = vf.verify(pkg, run_episode(update_task, _update_actions(update_task)))
    assert result.status == "pass", result.evidence


def test_missing_update_op_fails_update_miss(catalog, update_task):
    pkg = pkg_for(update_task, catalog, seed=41)
    result = vf.verify(pkg, run_episode(update_task, [{"op": "done"}]))
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "update_miss"]
    detail = result.evidence[-1]
    assert "profile_diff" in detail and "allergies" in detail["profile_diff"]


def test_missing_log_op_fails_log_miss(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, [{"op": "done"}]))
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "log_miss"]


# --------------------------------------------------------------------------- #
# plan boundary tests (recommend)
# --------------------------------------------------------------------------- #


def _fitting(rec_task):
    plan = fitting_plan(
        rec_task.s0.catalog, rec_task.oracle.plan_windows,
        rec_task.oracle.profile.allergies,
    )
    assert plan, "no fitting plan for the fixture"
    return plan


def _submit(plan):
    return [{"op": "submit_plan", "items": plan}, {"op": "done"}]


def _total(catalog, plan, nutrient):
    return sum(
        float((catalog[i["food_id"]].get("nutrients") or {}).get(nutrient, 0.0))
        * float(i["grams"]) / 100.0
        for i in plan
    )


def test_two_different_valid_plans_both_pass(catalog, rec_task):
    plan = _fitting(rec_task)
    pkg = pkg_for(rec_task, catalog, seed=12)

    first = vf.verify(pkg, run_episode(rec_task, _submit(plan)))
    assert first.status == "pass"

    # order-independent: reversed multiset is a different sequence, same pass
    second = vf.verify(pkg, run_episode(rec_task, _submit(list(reversed(plan)))))
    assert second.status == "pass"

    # a genuinely different multiset: add grams while staying inside every hi
    head = dict(plan[0])
    per_gram = {
        n: float((catalog_i := rec_task.s0.catalog[head["food_id"]])
                 .get("nutrients", {}).get(n, 0.0)) / 100.0
        for n in rec_task.oracle.plan_windows
    }
    headroom = min(
        (hi - _total(rec_task.s0.catalog, plan, n)) / per_gram[n]
        for n, (lo, hi) in rec_task.oracle.plan_windows.items()
        if per_gram[n] > 0
    )
    if headroom >= 5.0:  # enough slack to build a different in-window plan
        head["grams"] = head["grams"] + 5.0
        variant = [head] + [dict(item) for item in plan[1:]]
        third = vf.verify(pkg, run_episode(rec_task, _submit(variant)))
        assert third.status == "pass", third.evidence


def test_window_edge_just_inside_and_outside(catalog, rec_task):
    """Pin the window boundary exactly: total == hi passes, hi + epsilon fails."""
    plan = _fitting(rec_task)
    kcal_total = _total(rec_task.s0.catalog, plan, "kcal")

    inside = dataclasses.replace(
        rec_task,
        oracle=dataclasses.replace(
            rec_task.oracle,
            plan_windows={**rec_task.oracle.plan_windows,
                          "kcal": (kcal_total - 0.5, kcal_total)},
        ),
    )
    pkg = pkg_for(inside, catalog, seed=12)
    result = vf.verify(pkg, run_episode(inside, _submit(plan)))
    assert result.status == "pass", result.evidence

    outside = dataclasses.replace(
        rec_task,
        oracle=dataclasses.replace(
            rec_task.oracle,
            plan_windows={**rec_task.oracle.plan_windows,
                          "kcal": (kcal_total - 0.5, kcal_total - 0.5)},
        ),
    )
    pkg_out = pkg_for(outside, catalog, seed=12)
    result_out = vf.verify(pkg_out, run_episode(outside, _submit(plan)))
    assert result_out.status == "fail"
    assert result_out.failure_codes == ["task_fail", "window"]
    misses = result_out.evidence[-1]["window_misses"]
    assert any(m.startswith("kcal ") and "outside" in m for m in misses), misses


def test_allergen_in_plan_fails_allergy(catalog, rec_task):
    """An allergen-carrying food in the plan → fail(allergy), with the
    offending food in evidence."""
    profile = rec_task.oracle.profile
    offenders = [
        fid
        for fid, entry in rec_task.s0.catalog.items()
        if profile.allergies and profile.allergies[0] in (entry.get("allergen_tags") or [])
    ]
    if not offenders:  # the gym person has no allergies — pick a milk-allergic one
        pytest.skip("fixture person is allergy-free; allergy covered by unit below")
    bad_plan = [{"food_id": offenders[0], "grams": 50.0}]
    pkg = pkg_for(rec_task, catalog, seed=12)
    result = vf.verify(pkg, run_episode(rec_task, _submit(bad_plan)))
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "allergy"]


def test_allergy_fail_with_known_allergen(catalog):
    """train-cleo (milk allergy) submits a milk-tagged food → fail(allergy)."""
    person = next(p for p in TRAIN_ROSTER if "milk" in p.allergies)
    task = generate_one(
        catalog=catalog, family="recommend", person=person, seed=12,
        occasion="dinner", shell="rec-dinner",
    ).accepted
    assert task is not None
    milk_food = next(
        fid for fid, entry in catalog.items() if "milk" in (entry.get("allergen_tags") or [])
    )
    pkg = pkg_for(task, catalog, seed=12)
    result = vf.verify(
        pkg, run_episode(task, _submit([{"food_id": milk_food, "grams": 50.0}]))
    )
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "allergy"]
    assert result.evidence[-1]["allergy_offenders"][0]["food_id"] == milk_food


def test_off_allowed_food_ids_fails_inventory_miss(catalog, rec_task):
    plan = _fitting(rec_task)
    allowed = {item["food_id"] for item in plan}
    stranger = next(
        fid
        for fid, entry in rec_task.s0.catalog.items()
        if fid not in allowed
        and entry.get("category") != rec_task.s0.catalog[plan[0]["food_id"]].get("category")
    )
    restricted = dataclasses.replace(
        rec_task,
        oracle=dataclasses.replace(rec_task.oracle, allowed_food_ids=sorted(allowed)),
    )
    sneaky = [{"food_id": stranger, "grams": 50.0}] + [dict(i) for i in plan]
    pkg = pkg_for(restricted, catalog, seed=12)
    result = vf.verify(pkg, run_episode(restricted, _submit(sneaky)))
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "inventory_miss"]
    assert result.evidence[-1]["off_allowed_food_ids"] == [stranger]


def test_nonexistent_food_id_fails_wrong_goal(catalog, rec_task):
    """False-positive guard: a made-up food_id can never pass."""
    bogus = [{"food_id": "99999999", "grams": 100.0}]
    pkg = pkg_for(rec_task, catalog, seed=12)
    result = vf.verify(pkg, run_episode(rec_task, _submit(bogus)))
    assert result.status == "fail"
    assert result.failure_codes == ["task_fail", "wrong_goal"]
    assert result.evidence[-1]["nonexistent_food_ids"] == ["99999999"]


# --------------------------------------------------------------------------- #
# indeterminate triggers
# --------------------------------------------------------------------------- #


def test_teacher_error_is_indeterminate(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    episode = run_episode(log_task, _log_actions(log_task), error="api timeout")
    result = vf.verify(pkg, episode)
    assert result.status == "indeterminate"
    assert result.execution == "error"
    assert result.scorer is None
    assert result.reward is None
    assert result.failure_codes == ["teacher_error"]


def test_teacher_no_finish_is_indeterminate(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    actions = _log_actions(log_task)[:-1] + [{"op": "get_profile"}]
    result = vf.verify(pkg, run_episode(log_task, actions))
    assert result.status == "indeterminate"
    assert result.failure_codes == ["teacher_no_finish"]


def test_teacher_invalid_op_is_indeterminate(catalog, log_task):
    """Built purely from trajectory metadata: the harness executed a fallback
    get_profile the assistant text never asked for (spec §12 legality)."""
    actions = _log_actions(log_task)
    actions[0] = {"op": "get_profile"}
    raw = {0: "I think maybe something? no json"}  # unparsable text
    episode = run_episode(log_task, actions, raw_overrides=raw)
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, episode)
    assert result.status == "indeterminate"
    assert result.execution == "invalid_op"
    assert result.failure_codes == ["teacher_invalid_op"]
    assert result.evidence[-1]["failing_turn"] == 0


def test_teacher_invalid_op_when_parse_differs_from_executed(catalog, log_task):
    """Text parses cleanly but to a DIFFERENT op than the one executed."""
    actions = _log_actions(log_task)
    actions[0] = {"op": "get_profile"}
    raw = {0: '{"op": "log_meal", "food_id": "1", "grams": 5}'}  # genuine parse ≠ executed
    episode = run_episode(log_task, actions, raw_overrides=raw)
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, episode)
    assert result.execution == "invalid_op"
    assert result.status == "indeterminate"


def test_oracle_error_is_indeterminate_never_fail(catalog, log_task, monkeypatch):
    """Scorer raises → oracle_exec=error, traceback in evidence, never fail."""

    class ExplodingScorer:
        def score(self, end_state, oracle):
            raise RuntimeError("boom: scorer exploded")

    monkeypatch.setattr(vf, "Scorer", ExplodingScorer)
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, _log_actions(log_task)))
    assert result.status == "indeterminate"
    assert result.oracle_exec == "error"
    assert result.failure_codes == ["oracle_error"]
    assert any("traceback" in str(item).lower() for item in result.evidence)
    assert result.reward is None


def test_env_reconstruction_mismatch_is_indeterminate(catalog, log_task, rec_task):
    """Verifying package A against task B's episode → env_mismatch."""
    pkg = pkg_for(log_task, catalog, seed=30)
    wrong_episode = run_episode(rec_task, _submit(_fitting(rec_task)))
    result = vf.verify(pkg, wrong_episode)
    assert result.status == "indeterminate"
    assert result.oracle_exec == "env_mismatch"
    assert result.failure_codes == ["env_reconstruction_mismatch"]
    assert any(isinstance(item, dict) and "check" in item for item in result.evidence)


def test_catalog_sha_mismatch_is_env_mismatch(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    tampered = dataclasses.replace(
        pkg,
        catalog=dataclasses.replace(pkg.catalog, catalog_sha="f" * 64),
    )
    result = vf.verify(tampered, run_episode(log_task, _log_actions(log_task)))
    assert result.status == "indeterminate"
    assert result.failure_codes == ["env_reconstruction_mismatch"]
    checks = [item for item in result.evidence if isinstance(item, dict)]
    assert any(item.get("check") == "catalog_sha" for item in checks)


def test_corrupt_environment_is_oracle_error(catalog, log_task):
    """A package whose environment block cannot reconstruct → oracle_exec=error."""
    pkg = pkg_for(log_task, catalog, seed=30)
    broken = dataclasses.replace(
        pkg,
        environment=dataclasses.replace(pkg.environment, s0={"profile": "garbage"}),
    )
    result = vf.verify(broken, run_episode(log_task, _log_actions(log_task)))
    assert result.status == "indeterminate"
    assert result.oracle_exec == "error"
    assert result.failure_codes == ["oracle_error"]


def test_gate_unachievable_never_reaches_verify(catalog, log_task):
    """gate.unachievable is routed indeterminate at the GATE stage (ticket
    005) — the verifier never sees an unachievable task. Asserted here so the
    §19.2 trigger list is complete: gates reject first, verify's vocabulary
    has no gate.* codes."""
    exam_ctx = gates.GateContext.from_exam(load_exam())
    assert gates.run(log_task, exam_ctx).keep is True  # sanity: fixture passes gates
    assert not any(
        code.startswith("gate.")
        for code in ("teacher_error", "teacher_no_finish", "teacher_invalid_op",
                     "oracle_error", "env_reconstruction_mismatch")
    )


# --------------------------------------------------------------------------- #
# composite (3-leg) verification
# --------------------------------------------------------------------------- #


def _three_leg(catalog):
    task, reason = fx.assemble_three_leg(catalog, seed=101, allergen="fish")
    assert task is not None, reason
    return task


def test_three_leg_composite_pass(catalog):
    task = _three_leg(catalog)
    pkg = pkg_for(task, catalog, seed=101, steps=("update", "log", "recommend"))
    result = vf.verify(pkg, run_episode(task, fx.replay_actions(task)))
    assert result.status == "pass", result.evidence
    assert result.failure_codes == []
    tags = [e for e in result.evidence if isinstance(e, dict) and "sub_tags" in e]
    assert tags and len(tags[0]["sub_tags"]) == 3


def test_three_leg_composite_fail_names_sub_oracle(catalog):
    """A composite with a broken recommend leg → fail, tag = first failing
    sub-oracle's tag, evidence carries the index and the concrete miss."""
    task = _three_leg(catalog)
    actions = fx.replay_actions(task)
    # sabotage the plan: grams scaled to blow the kcal window
    plan_action = next(a for a in actions if a.get("op") == "submit_plan")
    plan_action["items"] = [
        {**item, "grams": item["grams"] * 3} for item in plan_action["items"]
    ]
    pkg = pkg_for(task, catalog, seed=101, steps=("update", "log", "recommend"))
    result = vf.verify(pkg, run_episode(task, actions))
    assert result.status == "fail"
    assert result.failure_codes[0] == "task_fail"
    sub = [e for e in result.evidence if isinstance(e, dict) and "failing_sub_oracle" in e]
    assert sub and sub[0]["failing_sub_oracle"] == 2  # the recommend leg


# --------------------------------------------------------------------------- #
# diagnostic_scores never affects status/reward (§19.3)
# --------------------------------------------------------------------------- #


def test_diagnostic_scores_never_affect_status_or_reward(catalog, log_task):
    pkg = pkg_for(log_task, catalog, seed=30)
    result = vf.verify(pkg, run_episode(log_task, _log_actions(log_task)))
    assert result.status == "pass" and result.reward == 1.0
    # even a hand-set low soft score cannot flip the hard contract
    object.__setattr__(result, "diagnostic_scores", {"soft": 0.01})
    assert result.status == "pass" and result.reward == 1.0
    # and a fail keeps failing with a perfect soft score
    fail = vf.verify(pkg, run_episode(log_task, [{"op": "done"}]))
    object.__setattr__(fail, "diagnostic_scores", {"soft": 1.0})
    assert fail.status == "fail" and fail.reward == 0.0
