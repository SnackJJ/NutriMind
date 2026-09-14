---
id: 006
title: Arm startup assertions + manifest provenance
status: CLOSED (2026-09-11)
depends_on: [004, 005]
spec: ../spec.md
spec_sections: ["D9"]
---

# 006 — Arm assertions + manifest

**What to build:** On startup an arm prints and asserts: reward version,
reference-model revision, task-selection policy and band, advantage estimator,
rollout k, `parallel_tool_calls`, exam revision. Mismatch aborts before rollout
spend. Metrics go to a manifest with the same provenance fields the Data Factory
uses, plus `effective-gradient fraction`.

**Blocked by:** 004, 005.

**Status:** ready-for-agent

- [x] a mismatched configuration aborts with zero rollouts issued
- [x] a matching configuration emits the asserted fields
- [x] the manifest is diffable against a factory `run_manifest.json` provenance
      subset
