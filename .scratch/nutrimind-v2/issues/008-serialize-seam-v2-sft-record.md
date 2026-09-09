---
id: 008
title: serialize seam (Seam 4) → v2 SFT record (segments + train_on, plan truncation)
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] pure; a scripted Pass episode → exactly one record with
      `len(messages) == len(segments) == len(train_on)`, `segments[-1] == "final"`,
      `train_on` true only on `step` / `final`
- [ ] `meta` version block complete: `oracle_version`, `rubric_version`, `reward_version`,
      `environment_version`, `task_schema_version`, `catalog_sha`, `nutrienv_rev`,
      `nutrimind_rev`, `seed`, family / steps / tier / persona / batch
- [ ] each serialize-edge failure has its own test (no system turn; consecutive assistant;
      missing observation; empty; last turn not FINISH; record over `max_seq_tokens` →
      `too_long`)
- [ ] one turn without `reasoning_content` → `plan=""` + `meta.n_turns_without_plan == 1`;
      every turn without a plan → `serialize.no_plan_any_turn`
- [ ] `reasoning_content` longer than `plan_max_tokens` → truncated, `meta.plan_truncation`
      set, `op_json` still parses
- [ ] no `<tool_call>` / `<think>` / `<|im_start|>` in any assistant content
- [ ] if v2's own parse of `raw_action_text` does not yield the executed op, no record is
      produced
