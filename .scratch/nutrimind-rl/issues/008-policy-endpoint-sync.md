---
id: 008
title: Checkpoint → policy_spec endpoint (veRL rollout glue)
status: CLOSED (2026-09-11)
depends_on: [007]
spec: ../spec.md
spec_sections: ["D3", "D10"]
adr: [../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md]
---

# 008 — Policy endpoint sync

**What to build:** The missing glue: a veRL training step's current weights become
the `policy_spec` that `student_rollout` calls (URL, model id, tokenizer / chat
template identical to ticket 002). Record the mechanism (colocated vLLM, weight
sync cadence) in the ticket close notes — it is not a second seam: the driver
still only takes `policy_spec`.

**Blocked by:** 007.

**Status:** ready-for-agent

- [x] after a train step, a rollout issued through D3 sees the updated policy
      (tested at smoke scale, not full exam)
- [x] tokenizer and chat template used at generate-time match ticket 002
- [x] no second injection point besides `policy_spec`
- [x] close notes name the exact veRL rollout engine and sync cadence used


## Close notes

veRL rollout engine: `verl.workers.rollout.vllm_rollout.vLLMRollout` (colocated vLLM). Sync cadence: `every_train_step` (`src/training/rl/policy_sync.py`). `student_rollout` still takes only `policy_spec`.
