# NutriMind v2.0 — Single-query speech and query-budget pilot

Status: **ready-for-agent**
Tracker: local Markdown (`docs/agents/issue-tracker.md`)
Domain vocabulary: `CONTEXT.md` (repo root) — used verbatim below
Governing ADRs: [ADR-010](../../docs/decisions/010-nutrimind-v2-rescope.md),
[ADR-011](../../docs/decisions/011-batch1-sft-trajectory-short-plan-thinking-teacher.md)
(protocol half superseded by ADR-014),
[ADR-012](../../docs/decisions/012-nutrienv-read-only-benchmark.md),
[ADR-013](../../docs/decisions/013-sft-failure-recovery-coverage.md),
[ADR-014](../../docs/decisions/014-native-tool-calling-v2-protocol.md),
[ADR-015](../../docs/decisions/015-v2-post-training-stack-trl-sft-verl-rl.md)
Sibling specs: `.scratch/nutrimind-v2/spec.md` (Data Factory),
`.scratch/nutrimind-rl/spec.md` (RL)
Design notes: `docs/research/single-query-contextual-expander.md`,
`docs/research/post-training-query-budget-pilot.md`

> **This spec does not replace Batch-1.** The factory's production target remains
> ≈ 420 accepted Pass traces at the design §7 family mix (ticket
> `nutrimind-v2/020`). The 100 / 200 / 0 unique-query counts are a **pilot query
> budget** overlay. The production factory config is not edited by this spec.

---

## Problem Statement

Factory **speech** is a single LLM pass from a code **intent** to `{query, foods}`.
Today that pass can be fed a pool of catalog-shaped fields and asked to verbalize
them. The result is repetitive, catalog-like wording, and situational clues that a
real user would put in one sentence ("I stopped by the market this morning… for
dinner") are not a first-class input. When the query is still ambiguous across
catalog variants, **Pass** is left to the Scorer to guess — which is the wrong
layer.

Separately, Batch-1 ≈ 420 is a production corpus target. It is not a statement of
how many **unique query identities** an SFT cold start or an RL pilot needs. Mixing
those numbers with teacher `k`, GRPO group size, rollout count, optimizer steps, or
token budget makes a 100-query feasibility run look like a spec violation, or a
noisy training curve look like a data shortage.

A staged reward / credit-assignment roadmap is deliberately not specified here.
The verifier and binary reward already exist.

## Solution

Keep NutriMind tasks as **single query**. "Context" means in-query situation, not
a user-dialogue history. The authoring program chooses the canonical entity and
oracle semantics in code, then hands the **expander** a **semantic brief**. After
speech, a **query/entity consistency** check uniquely binds the utterance or
regenerates / rejects it. The Scorer never disambiguates catalog entities.

Count **unique query identities** as their own budget, independent of traces,
teacher attempts, group size, steps, and tokens. A pilot overlay uses 100 unique
queries for the SFT cold start, ~200 for the RL pool, and 0 for OPD until
student-induced failure states exist. Expand the RL pool only when diagnostics
show the current budget cannot carry a stable signal. Do not change the
production factory family mix to do this.

Reward stays the existing tri-state projection: Pass = 1, Fail = 0,
Indeterminate excluded.

## User Stories

1. As the maintainer, I want every factory task to remain one user utterance, so that in-query situation is not mistaken for a multi-turn user dialogue.
2. As the maintainer, I want situation cues (market vs restaurant, breakfast vs dinner, cooked at home vs ordered) to live inside that utterance, so that the expander can disambiguate without catalog jargon.
3. As the maintainer, I want the canonical food/entity chosen in code before speech, so that the expander does not invent or vote on identity.
4. As the maintainer, I want the expander to receive a **semantic brief**, so that it writes natural wording instead of concatenating catalog fields.
5. As the maintainer, I want the brief to name situation, time/meal, source, persona, and family intent, so that the utterance has enough to bind uniquely.
6. As the maintainer, I want the brief to omit internal IDs and unused catalog attributes, so that speech does not read like a database row.
7. As a test author, I want to assert that the LLM-facing payload is a brief rather than a field dump, so that a regression back to mechanical language is visible.
8. As the maintainer, I want the expander to preserve the intent's amount path and meal semantics, so that `resolve_portion` can still back-bind.
9. As the maintainer, I want the expander not to invent foods, quantities, preparations, or venues, so that oracle semantics stay under code control.
10. As the maintainer, I want wording and rhythm to vary across intents, so that the corpus is not a template with swapped nouns.
11. As the maintainer, I want a **query/entity consistency** check after speech, so that a natural but unbound query never reaches the gate as if it were solved.
12. As the maintainer, I want a query that can bind to more than one canonical catalog entity to be regenerated or rejected, so that the Scorer is never asked to guess.
13. As the maintainer, I want query wording that conflicts with the intent's meal, amount, preparation, or venue to be dropped, so that speech cannot silently rewrite the oracle.
14. As the maintainer, I want query vs structured `foods` disagreement to reject the draft, so that the two halves of speech cannot diverge.
15. As the maintainer, I want foods outside the allowed semantic binding to reject the draft, so that an unselected variant cannot sneak in.
16. As the maintainer, I want missing disambiguation, when several catalog variants remain possible, to reject the draft, so that "sounds fine" is not a pass.
17. As the maintainer, I want regeneration to consume the existing expander retry budget and then reject, so that a stuck intent does not loop.
18. As the maintainer, I want new author-stage failure codes for these rejects, so that the histogram separates bind failure from consistency failure.
19. As a test author, I want a table of ambiguous, mismatched, and conflicting utterances, so that each reject reason is pinned without a network.
20. As the maintainer, I want the injected expander contract `generate_one` already uses to stay satisfied, so that authoring does not patch the lab.
21. As the maintainer, I want the synthetic expander to keep working for offline tests, so that brief-aware production wrapping does not break CI.
22. As the maintainer, I want multiple surface realizations, when sampled, ranked by consistency then unique binding then diversity then exam/query collision, so that selection is constrained rather than "pick the longest catalog paraphrase".
23. As the maintainer, I want **unique query identity** counted in the run manifest, so that query budget is not confused with accepted-trace count.
24. As the maintainer, I want unique-query count kept separate from teacher `k`, GRPO group size, rollout count, optimizer steps, and token budget, so that each knob can move without relabeling the others.
25. As the maintainer, I want an SFT cold-start overlay of 100 unique queries, coverage-balanced across families, so that protocol feasibility can be tried without claiming Batch-1 is done.
26. As the maintainer, I want that overlay to leave the production factory family `target_n` values untouched, so that ticket `nutrimind-v2/020` still means ≈ 420 accepted Pass.
27. As the maintainer, I want teacher `k = 6` to remain a teacher-attempt setting, not a query-budget setting, so that 100 unique queries can still use six attempts each.
28. As the maintainer, I want Batch-1 ≈ 421 accepted Pass to stay a production design target, so that the pilot cannot be read as a silent rescope of §7.
29. As the maintainer, I want SFT readiness judged on held-out queries, not SFT loss alone, so that a fluent next-token curve cannot green-light GRPO.
30. As the maintainer, I want legal tool-call / schema validity reported, so that a student that cannot speak the protocol is not sent to outcome-only RL.
31. As the maintainer, I want finish and no-finish rates reported, so that truncation is visible.
32. As the maintainer, I want execution and oracle-reconstruction health reported, so that env/oracle faults are not scored as Fail.
33. As the maintainer, I want non-zero Pass@1 on at least some families, so that the cold start is known to complete some tasks.
34. As the maintainer, I want per-family and per-composite-type coverage reported, so that a global pass rate cannot hide an empty wall family.
35. As the maintainer, I want recovery-positive success reported where applicable, so that ADR-013 remains a measured property.
36. As the maintainer, I want mixed-reward group rate at the planned GRPO group size, so that all-zero groups are visible before the RL loop.
37. As the maintainer, I want the cold start declared insufficient when trajectories are almost all invalid or no-finish, or groups are all-zero, so that GRPO is not started on a dead signal.
38. As the maintainer, I want the cold start declared usable for an RL pilot when the student can execute the protocol and occasionally Pass and Fail, so that mixed groups exist for a meaningful subset of queries.
39. As the maintainer, I want an RL unique-query pool of about 200, so that the first GRPO/DAPO arm has an order-of-magnitude starting set.
40. As the maintainer, I want a controlled overlap with SFT and a held-out unique-query set, so that train reward cannot be the only number.
41. As the maintainer, I want the RL pool to consume factory TaskPackages, so that this spec never authors its own tasks.
42. As the maintainer, I want binary reward only in this pilot: Pass = 1, Fail = 0, Indeterminate excluded, so that a staged reward scheme cannot sneak in unnamed.
43. As the maintainer, I want the RL unique-query pool expanded only when listed diagnostics fire, so that a noisy curve is not treated as a data shortage.
44. As the maintainer, I want expansion refused when the problem is difficulty band, verifier health, group size, or coverage rather than query count, so that the first response is not "add queries".
45. As the maintainer, I want OPD's unique-query budget to start at 0, so that OPD is not given a fake SFT-relative ratio.
46. As the maintainer, I want later OPD queries selected from student-induced failure states, so that the teacher is correcting something the student actually did.
47. As the maintainer, I want that selection to require an unambiguous teacher correction, verifier/teacher agreement, and no catalog/oracle ambiguity, so that OPD does not train on a disputed world.
48. As the maintainer, I want OPD queries to be a derived subset of the RL pool when they exist, so that OPD is not a third authoring program.
49. As a test author, I want the production factory config's family mix unchanged by the overlay tests, so that a pilot run cannot silently rewrite Batch-1.
50. As the maintainer, I want ticket `nutrimind-v2/020` to stay independently runnable, so that speech work and the production corpus are not the same ticket.
51. As the maintainer, I want Batch-1 production speech to be allowed to pick up the brief and the consistency check once those tickets close, so that the 420-trace run does not have to keep the mechanical expander.
52. As the maintainer, I want no third design document for staged reward or credit assignment in this round, so that verifier fields already on `VerificationResult` remain the source of truth.

## Implementation Decisions

1. **One primary seam.** This spec hangs off the existing factory authoring entry: `build` with an injected expander that already satisfies the `generate_one` contract `(pool, *, persona, family, amount_path) -> {query, foods}`. No second pipeline. No lab patch (ADR-012).
2. **Canonical entity in code, speech second.** Intent enumeration and `generate_one` still choose the entity and oracle. The expander does not search the catalog.
3. **Semantic brief is the LLM-facing payload.** A NutriMind wrapper, sitting in front of the live expander, derives the brief from the already-chosen entity plus intent fields (situation, time/meal, source, persona, family). The live model does not receive a mechanical list of catalog fields. The synthetic expander may ignore the brief and still return bindable speech for tests.
4. **Situation is in-query only.** Useful cues: where the food came from, preparation, time/meal, family intent, persona/style, a causal clause inside the same utterance. No stored user-dialogue history is introduced.
5. **Query/entity consistency is an author-stage check.** After speech (and after `resolve_portion` bind), reject or regenerate when: foods are outside the allowed binding; an unselected food or conflicting variant appears; required disambiguation is missing; wording conflicts with meal/amount/preparation/venue; query and `foods` disagree; the query can bind to multiple canonical entities; the query is natural but not deterministically resolvable. New `author.*` failure codes; histogram bucket separate from `author.unresolvable` bind failures.
6. **Regenerate then reject.** Consistency failures consume the expander's existing retry budget (`parse_retries + 1` attempts). Exhaustion → author reject. No infinite loop, no gate-level guess.
7. **Candidate ranking when several realizations are sampled.** Order: query/entity consistency, unique entity binding, naturalness/style diversity, exam and verbatim/semantic collision checks. This is selection, not a request to verbalize the catalog row.
8. **Unique query identity is a first-class counter.** The run manifest reports it. Accepted-trace count, teacher attempts, GRPO `G`, rollout `k`, optimizer steps, and token budget remain separate fields. Do not use one number as a proxy for another.
9. **Pilot overlay, not a yaml rewrite.** SFT cold start = 100 unique queries, coverage-balanced, not a full Batch-1 set. RL unique-query pool ≈ 200, with controlled SFT overlap and a held-out set. OPD unique-query budget = 0 initially. Production factory family `target_n` (composite 200 + 3-leg 40, recommend 71, evaluate 55, log 42, update 13; sum 421) is unchanged. Ticket `nutrimind-v2/020` is unchanged as the production run.
10. **Teacher `k = 6` is not the query budget.** A 100-unique-query cold start may still use six teacher attempts per task.
11. **SFT go/no-go is a report, not a loss threshold.** Measured on held-out unique queries before outcome-only GRPO: schema/tool-call legality, finish vs no-finish, execution and oracle-reconstruction health, Pass@1 by family, per-family and composite-type coverage, recovery-positive rate where applicable, mixed-reward group rate at the planned `G`. Insufficient: almost all invalid or no-finish, or all-zero groups. Usable for an RL pilot: protocol executes, some tasks complete, both Pass and Fail exist for a meaningful subset of query groups.
12. **RL expansion rule.** Start with binary reward only. Grow the unique-query pool only when: too few mixed-reward groups; high seed variance with no held-out improvement; some families or composite types have no usable groups; train reward rises but held-out Pass@1 does not; reward is dominated by a shortcut or verifier artifact. Do not expand because the training curve is noisy until difficulty, verifier health, group size, and coverage have been checked.
13. **OPD is derived later.** No OPD queries in the overlay. When OPD is configured, select from student-induced states where the teacher correction is unambiguous, verifier and teacher agree, the issue is not catalog/oracle ambiguity, and the state is a useful recovery or decision boundary. Those queries are a subset of the RL pool, not a ratio of SFT.
14. **Reward is not respecified.** `VerificationResult` plus `failure_codes` / `evidence` / `sub_tags` / `failing_sub_oracle` / `recovery_codes` stay as in the factory and RL specs. Reward map remains Pass=1.0, Fail=0.0, Indeterminate excluded (`reward_version = v2-r1`). A staged reward or credit-assignment roadmap is a future document, not a ticket in this feature.
15. **020 independence.** Speech tickets may close before or after `nutrimind-v2/020`. Production speech should consume the brief and the consistency check once they exist; 020 is not rewritten to wait. The pilot 100-query overlay is not a substitute for 020.

## Testing Decisions

Test external behaviour through the existing authoring seam. Prefer the highest seam: `build` with an injected expander (synthetic or a brief-recording double) and no network.

- Brief vs dump: the live expander's LLM-facing payload contains situation/persona/intent and a natural entity handle, and does not contain a mechanical catalog-field list. A recording double is enough.
- Single query: authored `Task.query` is one user utterance; no dialogue-history field is introduced.
- Consistency table: each reject reason in Implementation Decision 5 has a fixture that is dropped at author with the matching `author.*` code, and a uniquely binding utterance that is kept.
- Scorer isolation: an ambiguous two-variant query never becomes a gated TaskPackage.
- Unique-query counter: two speech realizations of the same identity count as one; two distinct identities count as two; teacher `k` does not inflate the counter.
- Overlay isolation: a pilot 100-unique-query run cannot change production family `target_n` in the production factory config.
- Go/no-go: a scripted all-invalid / all-no-finish set is insufficient; a scripted set with mixed Pass/Fail on more than one family is usable.
- RL expansion: a noisy-curve fixture with healthy mixed groups does not trigger expansion; a no-mixed-group fixture does.
- OPD: unique-query budget 0; a selector given a catalog-ambiguous state refuses it.

Prior art: `tests/training/data_factory/test_build.py` (injected synthetic expander), `test_author_widen.py`, `test_recovery.py`, `tests/training/rl/test_reward.py` (verifier projection). Do not freeze `generate_one` internals. Do not patch `../nutri-env-lab`.

**Seam count:** one new behaviour on the existing factory authoring seam. RL tickets consume TaskPackages and the existing reward / arm / difficulty seams. No third injection point for reward.

## Out of Scope

- Multi-turn user dialogue, stored chat history, or a conversational user simulator.
- Exposing canonical catalog IDs in the user-facing query.
- Replacing programmatic food binding or gates with prompt engineering.
- Editing the production factory config or the Batch-1 family mix / ticket 020 target.
- Treating 100 / 200 / 0 as a theoretical minimum or as a replacement for ≈ 420 accepted Pass.
- Staged reward, dense process reward, or a credit-assignment roadmap.
- Changing `VerificationResult` or `reward_version = v2-r1`.
- Running OPD (DAgger / GKD) in this feature; only the unique-query budget and later selection rule.
- Batch 2 shapes, full-length CoT, parallel tool calls, GiGPO, DPO, PPO.
- Patching NutriEnv (ADR-012).
- The v2 SFT trainer implementation (ADR-015; loader already exists).
- Declaring a v1.1 exam.

## Further Notes

Design notes remain the prose source for examples; this spec is the contract. If they drift, the spec wins.

Verifier fields already in code (`VerificationResult`, `failure_codes`, `evidence`, `sub_tags`, `failing_sub_oracle`, `recovery_codes`) are enough for the binary pilot. If a later round wants phased credit assignment, write a third design document then — not now.

### Vocabulary this spec uses (CONTEXT.md)

speech · expander · semantic brief · query/entity consistency · unique query identity ·
pilot query budget · intent · Pass-filter · trajectory · Pass · sweet-spot band ·
arm · recovery (ADR-013)
