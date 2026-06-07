# LLM-driven Kernel Engineering Landscape Maintainer

Data-driven toolkit for maintaining the **LLM Kernel Agent** field map as **Mermaid diagrams** synced into the root README.

```text
data/entries.yaml       ─┐
data/timeline.yaml       ├── validate → render_mermaid → sync_readme → README.md
data/timeline_dates.yaml ─┘
```

YAML is the source of truth; Mermaid blocks in README are generated — do not edit them by hand.

## What is included

```text
.
├── data/
│   ├── entries.yaml          # canonical papers / systems
│   ├── timeline.yaml         # curated report timeline buckets
│   ├── timeline_dates.yaml   # full-timeline date overrides + external labels
│   ├── mermaid_config.yaml   # README markers & diagram titles
│   ├── categories.yaml       # taxonomy (also used by category flowchart)
│   ├── sources.yaml
│   └── changelog.yaml
├── scripts/
│   ├── validate_entries.py
│   ├── render_mermaid.py     # YAML → .mmd
│   ├── sync_readme.py        # .mmd → README markers
│   ├── mermaid_lib.py
│   ├── render_landscape.py   # optional SVG export
│   ├── render_pptx.py
│   ├── render_drawio.py
│   └── fetch_candidates.py
├── figures/                  # generated .mmd (+ optional SVG/PPT)
└── docs/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make all
```

CI runs at repo root: `.github/workflows/render-landscape.yml` (validates + syncs both READMEs on push).

This validates data, writes locale-specific Mermaid to `figures/`, generates `README.zh-CN.md`, and updates both README files.

Generated Mermaid files:

```text
figures/landscape_timeline_report.en.mmd
figures/landscape_timeline_report.zh.mmd
figures/landscape_timeline_full.en.mmd
figures/landscape_timeline_full.zh.mmd
figures/landscape_category_map.en.mmd
figures/landscape_category_map.zh.mmd
```

## README languages

| File | Language |
|:-----|:---------|
| `../README.md` | English (default) |
| `../README.zh-CN.md` | Chinese |

Locale strings: `data/readme_locale.yaml`. Timeline titles: `data/mermaid_config.yaml`.

## Common commands

| Command | Action |
|:--------|:-------|
| `make validate` | Check entries + timeline references |
| `make mermaid` | Regenerate `.mmd` only |
| `make sync-readme` | Mermaid + patch README.md and README.zh-CN.md |
| `make all` | validate + mermaid + generate-readme-zh + sync-readme |
| `make svg` | Optional legacy SVG landscape |

## How to add a new paper

**1. Add to `data/entries.yaml`:**

```yaml
- id: new_kernel_agent
  name: NewKernelAgent
  year: 2026
  period: "2026+"
  category: Agent4Kernel
  subcategory: "Profiling-guided agent"
  backends: [CUDA, Triton]
  method_tags: [agent, profiling]
  source: awesome_llm_kernel_agent
  affiliation_short: "Your Lab · University"
  display_priority: 2
  show_in_report: false
  show_in_full: true
  timeline_date: "2026-06"   # optional; overrides timeline_dates.yaml
```

**2. (Optional) Add to curated timeline — `data/timeline.yaml`:**

```yaml
  - date: "2026-06"
    items:
      - id: new_kernel_agent
```

**3. (Optional) Set full-timeline date — `data/timeline_dates.yaml`:**

```yaml
overrides:
  new_kernel_agent: "2026-06"
```

**4. Regenerate:**

```bash
make all
```

## Three views

| View | Source | README location |
|:-----|:-------|:----------------|
| **Report timeline** | `timeline.yaml` | Main landscape section |
| **Full timeline** | `entries.yaml` + `timeline_dates.yaml` | `<details>` collapsible |
| **Category map** | `entries.yaml` × categories | `<details>` collapsible |

## Optional candidate fetching

```bash
python scripts/fetch_candidates.py
```

Writes `data/candidates.yaml` for manual review only.
