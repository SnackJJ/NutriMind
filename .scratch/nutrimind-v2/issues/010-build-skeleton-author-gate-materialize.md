---
id: 010
title: build skeleton — author→gate→materialize, --stop-after, reject routing, failure isolation
status: ready-for-agent
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

**Status:** ready-for-agent

- [ ] `build --stop-after gate` on a small intent set writes `intents/*.jsonl` +
      `tasks/*.jsonl` + `task_packages/*.json` and exits 0
- [ ] a catalog-SHA mismatch or a `nutrienv` rev mismatch aborts before any task is
      authored (non-zero exit, partial manifest written)
- [ ] one un-authorable intent among several good ones → a `rejects/author.jsonl` line, the
      good ones still materialize, the run completes
- [ ] a config error (missing key) fails the run immediately with non-zero exit
- [ ] intent enumeration is deterministic: two runs with the same config produce
      byte-identical `intents/*.jsonl`
- [ ] a re-run in the same `output_dir` skips any `task_id` already terminal (no `--force`)
- [ ] `expander` and `teacher_complete` are injected; nothing in core logic constructs them
