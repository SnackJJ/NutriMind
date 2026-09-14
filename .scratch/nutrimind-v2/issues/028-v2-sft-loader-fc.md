---
id: 028
title: v2 SFT loader for FC records (supersedes 019)
status: CLOSED (2026-09-11)
depends_on: [026]
spec: ../spec.md
spec_sections: ["9.2", "14.1", "14.2"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md, ../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md]
---

# 028 — v2 SFT loader (FC)

**What to build:** TRL `SFTTrainer` loader for §9.2 FC records. Applies the **student**
chat template with the **same** `tools` schema as eval. `train_on` drives the token
mask. Ticket **019 is SUPERSEDED** (it rejected `<tool_call>` as v1 and assumed
text-op `observation` segments).

**Blocked by:** 026.

**Status:** ready-for-agent

- [x] a valid §9.2 FC record tokenizes; labels are set on assistant turns with
      `train_on=true` (including rendered `tool_calls` and truncated
      `reasoning_content`)
- [x] `role=tool` / system / task turns have labels `-100`
- [x] a retired text-op record (`plan\n{"op":…}` and no `tool_calls`) is
      hard-rejected
- [x] a v1 XML record (`<tool_call>` / `<think>` / `<|im_start|>` as the action
      channel in `assistant.content`) is hard-rejected
- [x] the v1 loader is untouched and is never pointed at the v2 directory
