---
id: 010
title: build skeleton — author→gate→materialize, --stop-after, reject routing, failure isolation
status: CLOSED (2026-09-09) — 14 build tests green (328 dir-wide)
commit: 6dc5db9
depends_on: [004, 005, 006]
spec: ../spec.md
spec_sections: ["4.1", "4.3", "6", "8", "22.1", "US-2", "US-29"]
---

# 010 — build skeleton: author → gate → materialize

**What to build:** The thin `build(config, *, expander, teacher_complete)` orchestrator,
wired through **materialize** only (no teacher path yet). It:

1. Loads `configs/data_factory.yaml`; loads the catalog; asserts `catalog_digest(catalog)`
   equals the exam split's `catalog_sha256` **and** the installed `nutrienv` rev equals
   `config.nutrienv_rev` — either mismatch **aborts the run** (non-zero exit, partial
   `run_manifest.json`).
2. Enumerates **intents** per family: `(family, TRAIN_ROSTER person, seed, occasion,
   scene, shell/slots, amount_path, knife, steps, tier)` + a deterministic `task_id`,
   sorted by `task_id`, written to `intents/<family>.jsonl`.
3. Per intent: **author** (`generate_one` + injected `expander`, or the composition path
   later) → `gates.run` → `materialize` → `task_packages/<task_id>.json`.

`--stop-after {author,gate}` writes the staged artifact and stops (debug / staged
artifact, **not** a second pipeline). A single task failure records a
`rejects/{author,gate}.jsonl` line and the run continues. A config / schema /
un-importable-dependency error fails the whole run immediately.

**Blocked by:** 004, 005, 006.

**Status:** CLOSED (2026-09-09)

- [x] `build --stop-after gate` on a small intent set writes `intents/*.jsonl` +
      `tasks/*.jsonl` + `task_packages/*.json` and exits 0
- [x] a catalog-SHA mismatch or a `nutrienv` rev mismatch aborts before any task is
      authored (non-zero exit, partial manifest written)
- [x] one un-authorable intent among several good ones → a `rejects/author.jsonl` line, the
      good ones still materialize, the run completes
- [x] a config error (missing key) fails the run immediately with non-zero exit
- [x] intent enumeration is deterministic: two runs with the same config produce
      byte-identical `intents/*.jsonl`
- [x] a re-run in the same `output_dir` skips any `task_id` already terminal (no `--force`)
- [x] `expander` and `teacher_complete` are injected; nothing in core logic constructs them

## Closure notes

- `teacher_complete` is already a build() parameter (Seam-1 signature
  stability) but unused until ticket 011 wires the rollout stage after
  materialize.
- The gate-routing test monkeypatches `gates.run`: a REAL verbatim/
  semantic collision with the exam cannot be authored through
  `generate_one` (the expander's foods must be in the person's sampled
  pool, and brute-forcing the pool is impractical) — the gate itself is
  ticket 005's tested territory; build only routes its verdicts.
- `synth_expander` moved from tests/_fixtures.py to
  `src/training/data_factory/synthetic.py` (fixtures re-export it): the
  CLI needs an offline adapter and src must not import tests.
- The CLI refuses to run without an explicit `--expander` — a default
  would let a production run silently author synthetic speech. The
  production ark expander wrapper (structured {query, foods} JSON with
  parse retries) lands with ticket 012.
- Intent task_ids use the CANONICAL §10 task_key (config family
  `composite_update_log_recommend` → `composite--update+log+recommend--*`),
  and build asserts package.task_id == intent task_id — a mismatch means a
  bad enumerator and fails the whole run (§10).
- `--stop-after author` writes intents/ + tasks/ (gated: false) and stops;
  `--stop-after gate` additionally writes task_packages/ — per spec §4.1
  these are staged debug artifacts, not a second pipeline.
- Catalog mismatch tested with a tampered sqlite copy of the gold catalog
  (one food renamed → different digest, load_catalog still parses it).
