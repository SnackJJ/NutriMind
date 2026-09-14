---
id: 019
title: v2 SFT loader (minimal) — train_on → token mask after apply_chat_template; v1/v2 isolation
status: SUPERSEDED by 028 (2026-09-11) — §9.2 is FC; this ticket still rejects <tool_call> as v1 and trains the text-op shape
depends_on: [008]
spec: ../spec.md
spec_sections: ["9.2", "14.1", "14.2", "19.4", "22.10", "OQ-4", "OQ-5"]
---

# 019 — v2 SFT loader (minimal)

**What to build:** A v2-only SFT loader (module under `src/training/data_factory/`; exact
name is OQ-4 — non-blocking, renameable later). It reads v2 SFT records, applies the
**v2** chat template with the **v2** tokenizer, and for each message with `train_on[i]`
true sets `labels` on that message's content token span, everything else `-100` — the
token-level mask is **derived here, never stored in the record**.

It **hard-rejects** a record that: lacks `schema_version` / `segments` / `train_on`; has
`len(messages) != len(segments) != len(train_on)`; has `segments[-1] != "final"`; has a
`system` / `observation` message with `train_on = true`; or carries a `<tool_call>` /
`<think>` / `<|im_start|>` marker in any `assistant.content` (that is a v1 record).

The v1 loader (`src/training/sft/train.py`) is **not touched**. Out of scope: the v2
trainer itself. Scope confirmed with the maintainer (2026-09-09): build the minimal
loader; spec §14.2's "out of scope to build" is narrowed to the trainer.

**Blocked by:** 008.

**Status:** SUPERSEDED by 028 — do not implement this ticket. The v2 loader must accept FC records and reject the retired text-op shape.

- [ ] a valid v2 record → `(input_ids, labels)` where `labels` is `-100` on system /
      observation and equals the token ids on `step` / `final` content spans;
      `len(labels) == len(input_ids)`
- [ ] each spec §9.2 rejection rule has a test that trips it
- [ ] the v2 loader rejects a v1-shaped record; the v1 loader rejects a v2-shaped record
      (isolation test)
- [ ] no change to `src/training/sft/train.py` or any v1 loader file (empty diff for v1 paths)
- [ ] the tokenizer id and chat template are the loader's config, not read from the record
- [ ] the two loaders are never pointed at the same directory (asserted in the test setup)
