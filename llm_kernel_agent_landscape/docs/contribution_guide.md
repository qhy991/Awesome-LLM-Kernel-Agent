# Contribution Guide

## Add an entry

1. Edit `data/entries.yaml`.
2. Add a unique `id`.
3. Choose one valid `category` from `data/categories.yaml`.
4. Choose one valid `period` from `data/categories.yaml`.
5. Fill `backends`, `method_tags`, `source`, `affiliation_short`, and visibility flags.
6. Optionally set `timeline_date: "YYYY-MM"` for precise full-timeline placement.
7. If the work should appear in the **curated report timeline**, add it to `data/timeline.yaml`.
8. Run `make all` (validates, renders Mermaid, syncs README).

## Mermaid workflow

Do **not** hand-edit Mermaid blocks in the root README — they live between HTML comment markers and are overwritten by `sync_readme.py`.

| File | Purpose |
|:-----|:--------|
| `entries.yaml` | Canonical list of papers/systems |
| `timeline.yaml` | Curated report timeline (month buckets) |
| `timeline_dates.yaml` | Per-id date overrides + external labels for full timeline |
| `mermaid_config.yaml` | README marker names and diagram titles |

### Report timeline item formats

Reference an existing entry:

```yaml
- date: "2026-05"
  items:
    - id: fastkernels
```

Or add a free-form label (paper not yet in `entries.yaml`):

```yaml
    - label: "Kernel Contracts"
```

## Choose display flags

Use `show_in_report: true` for representative works in the legacy SVG map.

Use `show_in_full: true` for entries that should appear in the full Mermaid timeline and category map.

## Avoid clutter

The curated report timeline should stay readable (~3–5 items per month). Put secondary works only in `entries.yaml` + `timeline_dates.yaml`; they will appear in the **full timeline** automatically.

Use `affiliation_short` for on-figure SVG display (≤32 characters). Put full affiliation in optional `affiliation`.

## Highlight KernelOwl

```yaml
highlight: true
```

Only one or a few nodes should be highlighted (SVG export only).

## Manual review principle

`fetch_candidates.py` collects rough links from upstream awesome lists but never auto-edits `entries.yaml`.
