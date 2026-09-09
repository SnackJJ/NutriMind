---
id: 016
title: Compatibility guard (Seam 5) — public-API import + signature tests + ADR-012 two-class rule
status: ready-for-agent
depends_on: [003]
spec: ../spec.md
spec_sections: ["18", "19.6", "US-30"]
adr: [../../docs/decisions/012-nutrienv-read-only-benchmark.md]
---

# 016 — Compatibility guard (Seam 5)

**What to build:** The CI guard that protects **only** the public borrowed nutri-env API.

- `test_imports.py` — every symbol in the spec §18 "Public borrowed API" table resolves
  from its stated module.
- `test_borrowed_api_signatures.py` — `inspect.signature` assertions for that public set
  **only**. **No** signature assertion for any underscore-private helper
  (`_update_from_template`, `_bind_log_foods`, `_log_then_recommend`, `_parse_action`,
  `_SYSTEM_V2`, `split._item` / `_s0` / `_oracle`, …).
- a test asserting ADR-012's amended two-class rule matches what the guard files actually
  check: public → signature-guarded; private → not frozen (behaviour-tested elsewhere,
  via Seam 1).
- supersede / extend ticket 001's `test_nutrienv_smoke.py` where it overlaps rather than
  duplicating it.

**Blocked by:** 003. (Independent of the pipeline — can land early.)

**Status:** ready-for-agent

- [ ] every symbol in the §18 public table imports; a test fails loudly if one moves or
      disappears
- [ ] `inspect.signature` is asserted for the public set only; grepping the guard files for
      any underscore-private nutri-env name returns nothing
- [ ] a test encodes the two-class rule (public = signature-guarded; private = explicitly
      not, covered by behaviour tests instead)
- [ ] the guard runs in CI on the pinned `nutrienv` rev and is green
- [ ] no behaviour test lives in the guard files (Seam 5 is split from behaviour)
