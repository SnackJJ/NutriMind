---
id: 002
title: Train and eval prompts are token-identical for the same TaskPackage
status: CLOSED (2026-09-11)
depends_on: [001]
spec: ../spec.md
spec_sections: ["D2"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 002 — Train/eval prompt identity

**What to build:** The prompt construction used in student rollout (train) and in
exam/mini-exam eval is the same function. For one TaskPackage, the tokenized prompt
is byte-identical.

**Blocked by:** 001.

**Status:** ready-for-agent

- [x] system prompt and `tools` schema are the lab objects, not a second copy
- [x] a test asserts token identity for a frozen TaskPackage under train vs eval
      call sites
- [x] `parallel_tool_calls=false` is part of the asserted payload
