#!/usr/bin/env python
"""Convert combined Python scripts to Jupyter notebooks.

If a file uses percent-format markers (`# %%` / `# %% [markdown]`), preserve them
as notebook cells.

For plain scripts, infer cell boundaries from section dividers such as
`# ======` and split top-level function / main blocks into separate code cells so
plot types are easier to run independently in notebooks.
"""
import glob
import json
import os
import re

SECTION_DIVIDER_RE = re.compile(r'^\s*#\s*={6,}\s*$')
TOP_LEVEL_BLOCK_RE = re.compile(r'^(def |class |if __name__ == [\'"]__main__[\'"]:)')


def build_source(lines):
    source = ''.join(lines).strip('\n')
    if not source:
        return []
    src_lines = source.split('\n')
    formatted = [line + '\n' for line in src_lines[:-1]]
    if src_lines[-1]:
        formatted.append(src_lines[-1])
    return formatted


def make_cell(cell_type, lines):
    source = build_source(lines)
    if not source:
        return None
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def clean_section_title(line):
    title = line.lstrip().lstrip('#').strip()
    return ' '.join(title.split())


def is_section_header(lines, idx):
    if idx + 2 >= len(lines):
        return False
    return (
        SECTION_DIVIDER_RE.match(lines[idx].rstrip('\n'))
        and SECTION_DIVIDER_RE.match(lines[idx + 2].rstrip('\n'))
        and lines[idx + 1].lstrip().startswith('#')
    )


def split_top_level_blocks(lines):
    boundaries = []
    for idx, line in enumerate(lines):
        if line[:1] in {' ', '\t'}:
            continue
        if TOP_LEVEL_BLOCK_RE.match(line):
            boundaries.append(idx)

    if not boundaries:
        return [lines]

    chunks = []
    if ''.join(lines[:boundaries[0]]).strip():
        chunks.append(lines[:boundaries[0]])
    for chunk_idx, start in enumerate(boundaries):
        end = boundaries[chunk_idx + 1] if chunk_idx + 1 < len(boundaries) else len(lines)
        if ''.join(lines[start:end]).strip():
            chunks.append(lines[start:end])
    return chunks


def build_plain_script_cells(lines):
    cells = []
    section_lines = []
    current_title = None
    idx = 0

    def flush_section(title, body_lines):
        nonlocal cells
        if not ''.join(body_lines).strip():
            return
        if title:
            heading = make_cell("markdown", [f"## {title}\n"])
            if heading:
                cells.append(heading)
        for chunk in split_top_level_blocks(body_lines):
            cell = make_cell("code", chunk)
            if cell:
                cells.append(cell)

    while idx < len(lines):
        if is_section_header(lines, idx):
            flush_section(current_title, section_lines)
            current_title = clean_section_title(lines[idx + 1])
            section_lines = []
            idx += 3
            continue
        section_lines.append(lines[idx])
        idx += 1

    flush_section(current_title, section_lines)
    return cells

def convert(py_path, ipynb_path):
    with open(py_path) as f:
        lines = f.readlines()

    cells = []
    current_type = None
    current_lines = []
    leading_code_lines = []
    seen_percent_marker = False

    def flush():
        nonlocal current_type, current_lines
        if current_type is None:
            return
        source = build_source(current_lines)
        if not source:
            current_type = None
            current_lines = []
            return
        cell = {"cell_type": current_type, "metadata": {}, "source": source}
        if current_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        current_type = None
        current_lines = []

    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.startswith('# %% [markdown]'):
            if not seen_percent_marker:
                leading_cell = make_cell("code", leading_code_lines)
                if leading_cell:
                    cells.append(leading_cell)
                leading_code_lines = []
                seen_percent_marker = True
            flush()
            current_type = "markdown"
            current_lines = []
        elif stripped.startswith('# %%'):
            if not seen_percent_marker:
                leading_cell = make_cell("code", leading_code_lines)
                if leading_cell:
                    cells.append(leading_cell)
                leading_code_lines = []
                seen_percent_marker = True
            flush()
            current_type = "code"
            current_lines = []
        elif current_type == "markdown":
            if stripped.startswith('# '):
                current_lines.append(stripped[2:] + '\n')
            elif stripped == '#':
                current_lines.append('\n')
            else:
                current_lines.append(stripped + '\n')
        elif current_type == "code":
            current_lines.append(line)
        elif not seen_percent_marker:
            leading_code_lines.append(line)

    flush()

    # Fallback for plain .py scripts without percent-format cell markers.
    if not cells:
        cells = build_plain_script_cells(lines)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "mne", "language": "python", "name": "mne"},
            "language_info": {"name": "python", "version": "3.9.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(ipynb_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created {ipynb_path} with {len(cells)} cells")

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    for py in sorted(glob.glob(os.path.join(base, 'combined_*.py'))):
        ipynb = os.path.splitext(py)[0] + '.ipynb'
        convert(py, ipynb)
