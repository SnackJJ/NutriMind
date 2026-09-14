---
id: 018
title: RLVR export materializer (thin) — TaskPackage → rlvr/<task_id>.json
status: SUPERSEDED by 027 (2026-09-11) — prompt was react_manual("v2"); ADR-014 requires TOOL_SYSTEM_PROMPT + NUTRIENV_TOOLS
depends_on: [006, 010]
spec: ../spec.md
spec_sections: ["4.1", "4.2", "8", "9.3", "19.5"]
---

# 018 — RLVR export materializer (thin)

**What to build:** The `target=rlvr` branch of `build` — a **pure projection** of a
TaskPackage to `rlvr/<task_id>.json` (spec §9.3 schema), with **no teacher and no
trajectory**:

- `prompt` — `system` = `react_manual("v2")`, `task` = `"Task:\n<query>"`.
- `environment` — the same reconstruction block as the TaskPackage.
- `verifier` — `Scorer` reference + oracle payload + `oracle_version`.
- `reward` — adapter `binary`, `reward_version = v2-r1`,
  `{pass:1.0, fail:0.0, indeterminate:null}`.
- `termination` — `finish_ops` + `max_steps` matching the TaskPackage.
- `seed`, `meta` (provenance, versions).

An RLVR export is **never** derived from an SFT trajectory (spec §4.2). Tests are
schema-level (§19.5) — a full RLVR run is out of scope.

**Blocked by:** 006, 010.

**Status:** SUPERSEDED by 027 — do not implement this ticket. The §9.3 prompt is native tool calling, not `react_manual("v2")`.

- [ ] `build --target rlvr` writes `rlvr/<task_id>.json` for each gated task with zero
      teacher calls
- [ ] the export has `environment`, `verifier.oracle`, `reward.map`, `termination`; it has
      **no** `messages` and **no** teacher fields
- [ ] `environment` reconstructs to a runnable `NutriEnv` (same round-trip as the TaskPackage)
- [ ] `reward.map == {pass:1.0, fail:0.0, indeterminate:null}` with `reward_version == "v2-r1"`
- [ ] `termination.max_steps` and `finish_ops` equal the source TaskPackage's
- [ ] a test asserts the export path takes a TaskPackage as input and never an SFT record
