# Phase 2.5: SFT Data Scaling Pipeline (5000+ Queries)

## Overview
This document outlines the end-to-end architecture and execution plan for scaling our 192 high-quality seed queries into a diverse, production-ready pool of ~5000+ queries for Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

The pipeline uses varying techniques optimized for each Tier's unique complexity, leveraging both combinatorial logic and advanced LLM prompting (Evol-Instruct, CoT-Self-Instruct).

---

## 1. Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────┐
│  Candidate Seeds (~192 manually curated queries)         │
│  data/queries/candidate_seeds.json                       │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ T0 Seeds │   │ T1 Seeds │   │ T2 Seeds │   │ T3 Seeds │   │ T4 Seeds │   │ Err Seeds│
   │ ~22      │   │ ~60      │   │ ~45      │   │ ~50      │   │ ~28      │   │ ~15      │
   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
        │              │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼              ▼
   Direct QA      LLM Diverse    Evol-Instruct  Persona-based  Taxonomy-based Edge Case
   (No Tools)     Translation    (Tool combos)  Conditional    Safety Bounds  Recovery
   (Async API)    (Async API)    (Async API)    (Async API)    (Async API)    (Async API)
        │              │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼              ▼
   ~250 Qs        ~1200 Qs       ~1500 Qs       ~1750 Qs       ~800 Qs        ~280 Qs
        │              │              │              │              │              │
        └───────┬──────┴───────┬──────┴──────────────┴──────────────┴──────────────┘
                │              │
                ▼              ▼
            Formatting & JSONL Assembly
                │              │
                └──────┬───────┘
                       ▼
            Filtering & De-duplication
            (Cosine Sim > 0.85)
                       │
                       ▼
               ~5000 Query Pool
               (Ready for collect_trajectories.py)
```

---

## 2. Methodology Breakdown

### Step 1: T0 & T1 Scaling (QA and Factual Lookups)
**Target:** 250 T0 queries (4%) + 1200 T1 queries (21%)
**Strategy:** Base LLM API using randomized variable injection and strict prompt constraints.
*   **T0-qa:** General nutrition knowledge questions (e.g., "Why does the body need fiber?") where the model learns to answer *without* invoking tools. (Limited to 250 to prevent over-representing pure text generation).
*   **T1:** Simple factual lookups with varied language, foods, and metrics. 
*   **USDA Soft Constraint (Natural Error Spillover):** We explicitly DO NOT hard-restrict LLMs to only use the USDA vocabulary list. We provide a large sample of USDA items as *guidance/reference* in the prompt. 
    *   **Reasoning:** If a user queries "açaí bowl" or "Starbucks matcha latte" (not in our limited baseline DB), it correctly triggers a `food_not_found` error. The teacher model should then naturally generate an *Error Recovery Trajectory* (e.g., asking the user to use a generic ingredient name).
    *   This naturally supplements our Error dataset with highly realistic, organic failure cases rather than purely artificial ones.

*   **Prompting Approach**:
    ```xml
    Generate {N} single-step factual lookup queries about nutrition.
    Constraints to force diversity:
    1. Include the following foods: [salmon, quinoa, Big Mac...]
    2. Focus on these nutrients: [Potassium, Sugar...]
    3. Use varying tone: [casual, minor typos, slang, strict tracking format]
    ```
*   **Format**: Submitted via async LLM API calls alongside Step 2 & 3.

### Step 2: T2 Scaling (Evol-Instruct)
**Target:** 1500 queries (26%)
**Strategy:** Iterative scaling preventing mechanical memorization, with strict depth control.
T2 demands multi-step tool execution. To avoid the LLM memorizing a single rigid chain, evolution must enforce diverse tool combinations.
*   **Depth Restriction:** Evolution depth is STRICTLY limited to 1-2 steps to constrain complexity. Exceeding this boundary spills into T3 logic, wasting API calls and muddying the dataset.
*   **Evolution Methods (Tool Pairings)**:
    *   `get_food_nutrition` -> `log_meal` (Lookup then persist)
    *   `log_meal` -> `get_today_summary`
    *   `get_today_summary` -> `get_food_nutrition` (Check budget then calculate impact)
    *   `get_history` -> `retrieve_knowledge` (Trend spotting then fact checking)

### Step 3: T3 Scaling (Persona-based Conditional Logic)
**Target:** 1750 queries (30%)
**Strategy:** Conditional branching fueled by User Personas + State Injection.
T3 focuses on state-dependent and conditional reasoning. To prevent trajectory misalignment, we must generate matching system-level context.

*   **Persona Profile Pairing**: Instead of only generating the `query`, the LLM generates a matching `user_profile` JSON object. This ensures the condition implicitly contained in the query matches the "external" state the agent receives.
    ```json
    {
      "tier": "T3",
      "query": "I've been low on energy lately, should I increase my carbs?",
      "user_profile": {"age": 35, "gender": "female", "conditions": ["type_2_diabetes"]}
    }
    ```

### Step 4: T4 Scaling (Taxonomy-based Safety Bounds)
**Target:** 800 queries (15%)
**Strategy:** Strict taxonomy adherence.
T4 queries define our medical boundaries. LLMs left to "free ideate" will miss edge cases. We enforce generation across a rigid taxonomy:
1.  **Drug Interactions**: Warfarin, MAOIs, Statins.
2.  **Extreme Dieting**: Eating disorders, dangerous fasting.
3.  **Disease Self-Diagnosis**: Asking the agent to diagnose symptoms based on diet.
4.  **Replacing Professional Advice**: Post-surgery, transplant recovery protocols.

### Step 5: Robustness & Error Recovery Edge Cases
**Target:** ~280 queries (~5% of total pool)
**Strategy:** Intentional generation of tricky, ambiguous, or flawed queries to trigger fallback/recovery tools (Generated via async API using `error_recovery` seeds).
*   **Missing Quantities**: "Log my snack: I had some trail mix." (Forces clarification tool/response).
*   **Ambiguous Items/Commands**: "Search for apple, wait no, calculate a banana."
*   **Out-of-Vocabulary**: "How many calories in dragon-meat?" (Forces graceful failure handling).
*   **Constraint Violations**: Setting negative macro goals or impossible targets.

### Step 6: Quality Filtering & De-duplication
**Strategy:** Automated pruning pipeline coupled with Trajectory Reclassification.
*   **Natural Trajectory Reclassification** (New): When the Teacher Model runs the finalized queries, any T1 or T2 query that fails due to `food_not_found` will be **automatically reclassified as an Error Recovery query** in the final SFT dataset, teaching the model fallback behavior instead of discarding the entry.
*   **Exact Match / Keyword Overlap**: Fast upfront culling.
*   **Semantic De-duplication**: Embed all generated queries (using a lightweight model like `all-MiniLM-L6-v2` via `sentence-transformers` or a cloud vector embedding API). Remove queries with Cosine Similarity > 0.85.
*   **Rule Validation**: Ensure queries intended for pure-QA aren't actually asking for specific macro calculations, and that English ratio > 95%.
*   **Tier Classification Consistency (LLM-as-a-Judge)**: **CRITICAL STEP.** Run generated queries through a lightweight LLM classifier. If an evolved T2 query has drifted so far that the judge tags it as T3 (or if a T4 drifts to T3), drop it. This guarantees bucket purity.

---

## 3. Execution Checklist

- [x] Consolidate and validate the `candidate_seeds.json` baseline (192 seeds).
- [x] Export USDA food name list (`data/usda_foods.json`, 317 items) for T1 constraint injection.
- [ ] Implement `scripts/generate_queries.py` with async API calls and dynamic schemas (handling `user_profile` dict creation for T3).
- [ ] Run async API generation for Steps 1, 2, 3 & 4.
- [ ] Implement `scripts/deduplicate_queries.py` (Includes USDA fuzzy-match verifier and Cosine similarity).
- [ ] Merge into final `data/queries/final_sft_pool.jsonL`.
