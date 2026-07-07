#!/usr/bin/env python3
"""Generate README.zh-CN.md from README.md and locale strings.

Usage:
    python scripts/generate_readme_zh.py
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def replace_table_header(line: str, old: list[str], new: list[str]) -> str:
    result = line
    for src, dst in zip(old, new):
        result = result.replace(f"| {src} ", f"| {dst} ", 1)
        result = result.replace(f"| :{src}", f"| :{dst}", 1)
    return result


def extract_section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start = text.index(start_marker)
    if end_marker:
        end = text.index(end_marker, start)
        return text[start:end]
    return text[start:]


def translate_papers_block(en_readme: str, zh: dict) -> str:
    th = zh["table_headers"]
    papers_block = extract_section(en_readme, "## 📚 Research Papers", "## 📄 Citation")
    papers_block = papers_block.replace(
        "## 📚 Research Papers",
        f"## 📚 {zh['papers_title']}",
    )
    papers_block = papers_block.replace(
        "### 🤖 LLM-Based Kernel Generation Methods",
        f"### 🤖 {zh['methods_title']}",
    )
    papers_block = papers_block.replace(
        "> This section covers methods that use LLMs to automatically generate and optimize GPU kernels across various platforms.",
        f"> {zh['methods_desc']}",
    )
    papers_block = papers_block.replace(
        "<summary><b>📋 View All Methods (66 papers)</b></summary>",
        f"<summary><b>📋 {zh['methods_summary']}</b></summary>",
    )
    papers_block = re.sub(
        r"\| Title .* \|                                     Topics                                      \|",
        "| " + " | ".join(th["methods"]) + " |",
        papers_block,
        count=1,
    )
    papers_block = re.sub(
        r"\| :[-: ]+\| :[-: ]+\| :[-: ]+\| :[-: ]+\| :[-: ]+\| :[-: ]+\| :[-: ]+\| :[-: ]+\|",
        "| "
        + " | ".join(f":{'-' * max(3, len(h)-2)}" for h in th["methods"])
        + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "### 🧠 Agent Methodology & Techniques",
        f"### 🧠 {zh['agent_title']}",
    )
    papers_block = papers_block.replace(
        "> General agentic and reasoning techniques applicable to kernel generation and systems optimization.",
        f"> {zh['agent_desc']}",
    )
    papers_block = re.sub(
        r"\| Title .* \|                   Topics                    \|",
        "| " + " | ".join(th["agent"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "### 📊 Datasets and Benchmarks",
        f"### 📊 {zh['benchmarks_title']}",
    )
    papers_block = papers_block.replace(
        "> Evaluation benchmarks and datasets for assessing LLM-generated kernel quality and performance.",
        f"> {zh['benchmarks_desc']}",
    )
    papers_block = re.sub(
        r"\| Title .* \|             Platforms              \|",
        "| " + " | ".join(th["benchmarks"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "### 🛠️ Traditional Kernel Optimization",
        f"### 🛠️ {zh['traditional_title']}",
    )
    papers_block = papers_block.replace(
        "> Manual kernel optimization techniques, DSLs, and educational resources for understanding GPU programming fundamentals.",
        f"> {zh['traditional_desc']}",
    )
    papers_block = re.sub(
        r"\| Title .* \|                                  Topics & Resources                                   \|",
        "| " + " | ".join(th["traditional"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "## 🌟 Community & Resources",
        f"## 🌟 {zh['community_title']}",
    )
    papers_block = papers_block.replace(
        "### 🧰 Open-Source Tools & Frameworks",
        f"### 🧰 {zh['tools_title']}",
    )
    papers_block = papers_block.replace(
        "> Agentic kernel optimization systems and hands-on CUDA engineering resources.",
        f"> {zh['tools_desc']}",
    )
    papers_block = re.sub(
        r"\| Title .* \| Topics \|",
        "| " + " | ".join(th["tools"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "### 📰 Industry Blogs & News",
        f"### 📰 {zh['blogs_title']}",
    )
    papers_block = re.sub(
        r"\| Title .* \|                  Topics                   \|",
        "| " + " | ".join(th["blogs"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "### 🏆 Competitions & Challenges",
        f"### 🏆 {zh['competitions_title']}",
    )
    papers_block = re.sub(
        r"\| Event .* \|   Platform   \|",
        "| " + " | ".join(th["competitions"]) + " |",
        papers_block,
        count=1,
    )

    papers_block = papers_block.replace(
        "## 🤝 Contributing",
        f"## 🤝 {zh['contributing_title']}",
    )
    papers_block = papers_block.replace(
        "We welcome contributions! If you know of a paper, tool, or resource that should be included, please:",
        zh["contributing_lead"],
    )
    papers_block = papers_block.replace(
        "1. **Fork** this repository",
        f"1. {zh['contributing_steps'][0]}",
    )
    papers_block = papers_block.replace(
        "2. **Add** your entry following the existing format",
        f"2. {zh['contributing_steps'][1]}",
    )
    papers_block = papers_block.replace(
        "3. **Submit** a pull request with a brief description",
        f"3. {zh['contributing_steps'][2]}",
    )
    papers_block = papers_block.replace(
        "### Contribution Guidelines",
        f"### {zh['guidelines_title']}",
    )
    for en_line, zh_line in zip(
        [
            "- Ensure the paper/resource is relevant to LLM-based kernel generation or GPU optimization",
            "- Include proper citation with title, venue, date, and links",
            "- Add appropriate topic tags",
            "- Maintain chronological order (newest first)",
            "- Check for duplicates before submitting",
        ],
        zh["guidelines"],
    ):
        papers_block = papers_block.replace(en_line, f"- {zh_line}")

    return papers_block


def patch_english_landscape(content: str, en_loc: dict) -> str:
    content = content.replace(
        "GitHub 会在 README 中**原生渲染 Mermaid**。Landscape 由 [`llm_kernel_agent_landscape/`](llm_kernel_agent_landscape/) 维护：**改 YAML → 生成 Mermaid → 同步到 README**。",
        en_loc["landscape_intro"],
    )
    content = content.replace(
        "# validate + render + sync README",
        f"# {en_loc['landscape_cmd_comment']}",
    )
    content = content.replace("### 精选时间线", en_loc["curated_timeline"])
    content = content.replace(
        "<summary><b>完整时间线（全部 curated entries + 待收录项）</b></summary>",
        f"<summary>{en_loc['full_timeline_summary']}</summary>",
    )
    content = content.replace(
        "<summary><b>分类全景（5 大类别 flowchart）</b></summary>",
        f"<summary>{en_loc['category_map_summary']}</summary>",
    )
    content = content.replace("### 增量维护速查", en_loc["maintenance_heading"])
    content = content.replace("| 目标 | 编辑文件 |", "| Goal | Edit file |")
    content = content.replace(
        "| 新增论文 / 系统 | `data/entries.yaml` |",
        "| Add paper / system | `data/entries.yaml` |",
    )
    content = content.replace(
        "| 调整精选时间线 | `data/timeline.yaml` |",
        "| Adjust curated timeline | `data/timeline.yaml` |",
    )
    content = content.replace(
        "| 调整完整时间线日期 | `data/timeline_dates.yaml`（`overrides` / `external`） |",
        "| Adjust full-timeline dates | `data/timeline_dates.yaml` (`overrides` / `external`) |",
    )
    content = content.replace(
        "| 改 README 锚点 / 标题 | `data/mermaid_config.yaml` |",
        "| Change README markers / titles | `data/mermaid_config.yaml` |",
    )
    content = content.replace(
        "<summary><b>可选：YAML → SVG 分类地图（slides / PPT）</b></summary>",
        f"<summary>{en_loc['svg_optional_summary']}</summary>",
    )
    if en_loc["lang_link"] not in content:
        content = content.replace(
            "[📖 Introduction]",
            f"{en_loc['lang_link']}\n\n[📖 Introduction]",
            1,
        )
    return content


def build_chinese_landscape_block(en_readme: str, zh: dict) -> str:
    block = extract_section(en_readme, "## 🗺️ Landscape Map", "---\n\n## 📚 Research Papers")
    block = block.replace("## 🗺️ Landscape Map", f"## 🗺️ {zh['landscape_title']}")
    block = re.sub(
        r"GitHub \*\*natively renders Mermaid\*\*[^\n]+sync to README\*\*\.",
        zh["landscape_intro"],
        block,
        count=1,
    )
    block = block.replace(
        "# validate + render + sync README",
        f"# {zh['landscape_cmd_comment']}",
    )
    block = block.replace("### Curated Timeline", zh["curated_timeline"])
    block = block.replace(
        "<summary><b>Full timeline (all curated entries + pending items)</b></summary>",
        f"<summary>{zh['full_timeline_summary']}</summary>",
    )
    block = block.replace(
        "<summary><b>Category map (5-column flowchart)</b></summary>",
        f"<summary>{zh['category_map_summary']}</summary>",
    )
    block = block.replace("### Maintenance Quick Reference", zh["maintenance_heading"])
    mt = zh["maintenance_table"]
    block = re.sub(
        r"\| Goal \| Edit file \|\n\|:[^\n]+\|\n(?:\|[^\n]+\|\n)+",
        "\n".join(
            ["| " + " | ".join(mt[0]) + " |", "|:" + ":|:".join(["-" * 3] * len(mt[0])) + "|"]
            + ["| " + " | ".join(row) + " |" for row in mt[1:]]
        )
        + "\n",
        block,
        count=1,
    )
    block = block.replace(
        "<summary><b>Optional: YAML → SVG category map (slides / PPT)</b></summary>",
        f"<summary>{zh['svg_optional_summary']}</summary>",
    )
    return block


def main() -> int:
    locale_cfg = load_yaml(ROOT / "data" / "readme_locale.yaml")
    zh = locale_cfg["zh"]
    en_loc = locale_cfg["en"]
    en_readme = (REPO / "README.md").read_text(encoding="utf-8")

    papers_block = translate_papers_block(en_readme, zh)
    nav = zh["nav"]
    landscape_block = build_chinese_landscape_block(en_readme, zh)

    header = f"""<div align="center">

![Awesome LLM Kernel Agent Cover](assets/cover.png)

# 🚀 Awesome LLM Kernel Agent

### _{zh['subtitle']}_

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-May%202026-blue.svg)](https://github.com/qhy991/Awesome-LLM-Kernel-Agent)
[![Papers](https://img.shields.io/badge/Papers-100+-green.svg)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

{zh['lang_link']}

[📖 {nav['intro']}](#-{nav['intro']}) • [📚 {nav['papers']}](#-{zh['papers_title']}) • [🗺️ {nav['landscape']}](#-{zh['landscape_title']}) • [🎯 {nav['benchmarks']}](#-{zh['benchmarks_title']}) • [🧰 {nav['tools']}](#-{zh['tools_title']}) • [🛠️ {nav['resources']}](#-{zh['traditional_title']}) • [🤝 {nav['contributing']}](#-{zh['contributing_title']})

</div>

---

## 📖 {zh['intro_title']}

{zh['intro_lead']}

### 🎯 {zh['what_is_title']}

{zh['what_is_lead']}

"""
    for bullet in zh["bullets"]:
        header += f"- {bullet}\n"

    header += f"""
### 👥 {zh['who_title']}

"""
    for item in zh["who_items"]:
        header += f"- {item}\n"

    header += f"""
### 📊 {zh['stats_title']}

"""
    for stat in zh["stats"]:
        header += f"- {stat}\n"

    header += """
---

"""

    footer = f"""
## 📄 {zh['citation_title']}

{zh['citation_lead']}

```bibtex
@misc{{awesome-llm-kernel-agent,
  author = {{Haiyan Qin}},
  title = {{Awesome LLM Kernel Agent: A Curated Collection of LLM-Driven GPU Kernel Generation Research}},
  year = {{2025}},
  publisher = {{GitHub}},
  url = {{https://github.com/qhy991/Awesome-LLM-Kernel-Agent}}
}}
```

---

## 📜 {zh['license_title']}

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

{zh['license_text']}

---

<div align="center">

{zh['footer_star']}

{zh['footer_maintained']}

_{zh['footer_updated']}_

</div>
"""

    en_content = patch_english_landscape(en_readme, en_loc)
    (REPO / "README.md").write_text(en_content, encoding="utf-8")

    zh_readme = header + landscape_block + "\n---\n\n" + papers_block + footer
    out_path = REPO / "README.zh-CN.md"
    out_path.write_text(zh_readme, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
