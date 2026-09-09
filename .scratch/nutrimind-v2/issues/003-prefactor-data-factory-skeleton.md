---
id: 003
title: Prefactor — data_factory package skeleton, seam concept types, config schema, test wiring
status: ready-for-agent
depends_on: [002]
spec: ../spec.md
spec_sections: ["7", "14.2", "19.7", "OQ-4", "OQ-14"]
---

# 003 — Prefactor: data_factory skeleton, seam concept types, config schema, test wiring

**What to build:** A skeleton so every later slice only adds behaviour, never scaffolding.
The `src/training/data_factory/` package exists and imports with no side effects. The
shared seam vocabulary — the concept types the pure seams exchange — is defined once as
plain dataclasses / TypedDicts with **no logic**:

- `GateResult` (`keep`, `failure_code`, `reason_detail`, `stage`) — Seam 2 output.
- `TurnMeta` — one ReAct turn: `raw_action_text`, `executed_op`, `parse_status`,
  `fallback_used`, `fallback_reason`, `content`, `reasoning_content`, `finish_reason`,
  `usage`. **Produced by ticket 009, consumed by ticket 007** — so it lives here, and
  neither ticket depends on the other.
- `EpisodeResult` — one teacher rollout: `end_state` (the world state produced),
  `turns: list[TurnMeta]`, `reached_finish: bool`, `error: str | None`. This is the
  single input the verifier takes (Seam 3), replacing spec §19.7's looser
  `verify(task_package, end_state)` phrasing — `end_state` is now a field of it.
- `RolloutCache` — the on-disk `rollouts/cache/<task_id>.json` container:
  `task_id`, `attempts: list[{attempt_id, EpisodeResult, VerificationResult}]`,
  `selected_attempt: int | None`. Pinned here so ticket 011 writes it and ticket 014
  (`--from-stage serialize`) can re-read and re-select without re-running the teacher.
- `VerificationResult` — Seam 3 output, spec §12 shape.
- `TaskPackage` — the canonical artifact, spec §9.1 shape.

`configs/data_factory.yaml`
carries the full spec §7 key schema (`teacher`, `expander`, per-family
`{target_n, teacher_k, over_generate_x, amount_path_weights?, gram_anchor}`,
`max_seq_tokens`, `plan_max_tokens`, `tokenizer_name`, `max_intents`, `usd_budget`,
`on_budget`, `output_dir`, `rubric_version`, `reward_version`), keeps the existing
NutriEnv pin block untouched, and gets a loader that validates it (missing / mistyped key
→ a clear error). `pyproject.toml` gets `[tool.pytest.ini_options] testpaths=["tests"]`.
`.env.example` gains `ARK_API_KEY` / `ARK_BASE_URL` (OQ-14).

**Blocked by:** None — 001 and 002 are CLOSED prerequisites.

**Status:** ready-for-agent

- [ ] importing the `data_factory` package and every seam concept type (`GateResult`,
      `TurnMeta`, `EpisodeResult`, `RolloutCache`, `VerificationResult`, `TaskPackage`)
      succeeds with no network and no eager `nutrienv` import beyond what is already guarded
- [ ] the concept-type module has no logic — only field definitions and, at most, trivial
      `from_dict` / `to_dict`
- [ ] `configs/data_factory.yaml` has every spec §7 key; the config loader returns a typed
      object and raises a clear error on a missing or wrong-typed key (unit test)
- [ ] the NutriEnv `nutrienv:` pin block from ticket 001 is byte-unchanged
- [ ] `pytest` from the repo root discovers `tests/training/data_factory/` with no `-o` flags
- [ ] `.env.example` documents `ARK_API_KEY` and `ARK_BASE_URL`; no real key value committed
- [ ] no file under `src/training/sft/` or `src/training/grpo/` is modified
