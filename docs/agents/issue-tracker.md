# Issue Tracker

**Type**: Local Markdown (no GitHub Issues, no `gh issue create`).

Engineering skills (`to-spec`, `to-tickets`, `triage`, `qa`) read from and write to
Markdown files in this repo. Do **not** call `gh issue create` or create GitHub labels.

## Layout

Per-feature, under `.scratch/`:

```
.scratch/<feature>/
├── spec.md            ← the feature spec (from /to-spec)
└── issues/            ← one Markdown file per ticket (from /to-tickets)
```

Current feature:

```
.scratch/nutrimind-v2/
├── spec.md
└── issues/
```

## Triage labels

The `triage` skill is not in use for this feature. If it is later adopted, use the
canonical role names as label strings inside the ticket files' frontmatter:
`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.

## Domain docs

- Glossary: `CONTEXT.md` (repo root). Use its vocabulary; do not introduce synonyms.
- ADRs: `docs/decisions/` (**not** `docs/adr/`). Numbered `NNN-slug.md`.

`.gitignore` ignores `docs/*` except `docs/agents/` and `docs/decisions/`, so this file
and the ADRs are trackable. The rest of `docs/` (`plans/`, `specs/`, `handoff/`) stays
local-only. When first committing this exception, add the **whole** `docs/decisions/`
history (000–…, `historical-decisions.md`), not only the newest ADRs, so a fresh clone
has the complete architecture record.

## PRs as a request surface

Off. External PRs are not part of the triage queue.
