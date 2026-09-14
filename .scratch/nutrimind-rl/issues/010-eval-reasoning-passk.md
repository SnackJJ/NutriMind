---
id: 010
title: Eval — with-reasoning vs tools-only, pass@1 and pass@k
status: CLOSED (2026-09-11)
depends_on: [001, 005]
spec: ../spec.md
spec_sections: ["D8"]
adr: [../../docs/decisions/014-native-tool-calling-v2-protocol.md]
---

# 010 — Exam reporting

**What to build:** Student reported on the frozen v1.0 exam (ticket 005 gate)
both **with-reasoning** and **tools-only** (reasoning stripped, `tool_calls`
kept). pass@1 and pass@k together. Every number names the exam revision.

**Blocked by:** 001, 005.

**Status:** ready-for-agent

- [x] tools-only strips `reasoning_content` and does not change `tool_calls`
- [x] pass@1 and pass@k are both in the report
- [x] the report carries the exam revision; a dirty exam never produces a
      number (005)
