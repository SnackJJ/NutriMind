---
id: 008
title: serialize seam (Seam 4) → v2 SFT record (segments + train_on, plan truncation)
status: CLOSED (2026-09-09) — 18 serialize tests green (314 dir-wide)
commit: b0386a4
depends_on: [003, 006]
spec: ../spec.md
spec_sections: ["9.2", "11", "19.4", "22.10", "22.11"]
---

# 008 — serialize seam (Seam 4) → v2 SFT record

**What to build:** The pure record-writer.
`serialize(task_package: TaskPackage, episode: EpisodeResult) -> record` (both shared
types from ticket 003), producing the v2 SFT record (spec §9.2):

- OpenAI-shaped `messages`: `system` once, then the `Task:` user turn, then strictly
  alternating observation / assistant, last message an assistant turn whose op is in
  `nutrienv.harness.runner.FINISH_OPS`.
- parallel `segments` (`system` / `task` / `observation` / `step` / `final`) and
  per-message `train_on` (bool). Batch 1: `train_on[i]` true exactly where
  `segments[i] ∈ {step, final}`.
- `assistant.content = f"{plan}\n{op_json}"`. `plan` = teacher `reasoning_content` for
  that turn truncated to `plan_max_tokens` (token-exact with the injected student
  tokenizer if `tokenizer_name` set, else `~4 chars/token`; recorded in
  `meta.plan_truncation`). `op_json` = the op actually executed against `NutriEnv`.
- complete `meta` version block (spec §9.2).
- **No token-level `loss_mask` stored** — the v2 loader (ticket 019) derives it.

Serialize failure codes (spec §11): `serialize.empty_episode`, `.no_system_turn`,
`.consecutive_assistant`, `.missing_observation`, `.turn_count_mismatch`,
`.last_turn_not_finish`, `.too_long` (record tokens > `max_seq_tokens`),
`.no_plan_any_turn`. One missing `reasoning_content` is tolerated as `plan=""` +
`meta.n_turns_without_plan` counter. No `<tool_call>` / `<think>` / `<|im_start|>` marker
anywhere in assistant content.

**Blocked by:** 003, 006.

**Status:** CLOSED (2026-09-09)

- [x] pure; a scripted Pass episode → exactly one record with
      `len(messages) == len(segments) == len(train_on)`, `segments[-1] == "final"`,
      `train_on` true only on `step` / `final`
- [x] `meta` version block complete: `oracle_version`, `rubric_version`, `reward_version`,
      `environment_version`, `task_schema_version`, `catalog_sha`, `nutrienv_rev`,
      `nutrimind_rev`, `seed`, family / steps / tier / persona / batch
- [x] each serialize-edge failure has its own test (no system turn; consecutive assistant;
      missing observation; empty; last turn not FINISH; record over `max_seq_tokens` →
      `too_long`)
- [x] one turn without `reasoning_content` → `plan=""` + `meta.n_turns_without_plan == 1`;
      every turn without a plan → `serialize.no_plan_any_turn`
- [x] `reasoning_content` longer than `plan_max_tokens` → truncated, `meta.plan_truncation`
      set, `op_json` still parses
- [x] no `<tool_call>` / `<think>` / `<|im_start|>` in any assistant content
- [x] if v2's own parse of `raw_action_text` does not yield the executed op, no record is
      produced

## Closure notes

- `EpisodeResult` gained an optional `reset_observation` field (backward
  compatible; `from_dict` defaults None): the first user message of the
  serialized trajectory had no home in 003's types — each `TurnMeta.observation`
  is the obs that turn *produced*. `rollout` now records observations exactly
  as the base harness embeds them (`json.dumps` defaults, 6000-char cap) so
  the record's user messages are byte-faithful to what the teacher saw.
- The user message format is reconstructed: `Step budget: {max_steps - i}
  action(s) remaining, including this turn.\nObservation:\n{obs}` (matches
  the base `act()` and spec §9.2's example). max_steps derives from the family
  budget, consistent with `rollout`'s default.
- `validate_record` is exported — the loader-mirror structural check (the §9.2
  rejection rules). Serialize runs it on its own output; ticket 019's loader
  reuses it instead of re-implementing. Its `serialize.v1_marker` code is a
  v2-local extension (markers cannot survive serialize's sanitization).
- The invalid-op guard raises `serialize.invalid_op_turn` — defensive only:
  verify() already routes non-genuine parses to indeterminate/teacher_invalid_op
  (§12), so build never sends such episodes here.
- No `<tool_call>`/`<think>`/`<|im_start|>` markers in any assistant content:
  plans are sanitized at compose time (markers stripped; the text between
  them is kept — the spec forbids the markers, not the reasoning).
- Token counting for `too_long`: sum over messages (tokenizer-exact when
  injected, else ~4 chars/token rounded up per message).
