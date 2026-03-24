#!/usr/bin/env python

import csv
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
STATS_ROOT = OUTPUT_ROOT / "stats"
SLIDES_ROOT = OUTPUT_ROOT / "slides"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(path: Path, measure_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["measure"] = measure_name
            rows.append(row)
    return rows


def region_display_name(region: str) -> str:
    return region.replace("_", " - ")


def level_display_name(level: str) -> str:
    mapping = {
        "patient_level": "Patient Level",
        "trial_level": "Trial Level",
    }
    return mapping.get(level, level.replace("_", " ").title())


def sort_key(row):
    measure_order = {"Power": 0, "Coherence": 1}
    level_order = {"patient_level": 0, "trial_level": 1}
    return (
        measure_order.get(row["measure"], 99),
        region_display_name(row["region"]),
        level_order.get(row["level"], 99),
    )


def build_markdown(rows, slides_dir: Path) -> str:
    lines = [
        "---",
        "title: Retrieval MLMR Summary",
        "---",
    ]

    measure_dirs = {
        "Power": STATS_ROOT / "power",
        "Coherence": STATS_ROOT / "coherence",
    }

    for measure_name in ("Power", "Coherence"):
        measure_rows = [row for row in rows if row["measure"] == measure_name]
        overview_dir = measure_dirs[measure_name] / "overview"

        for level in ("patient_level", "trial_level"):
            overview_path = overview_dir / f"{level}_overview.png"
            if overview_path.exists():
                rel_image = os.path.relpath(overview_path, slides_dir)
                lines.extend(
                    [
                        "",
                        "---",
                        "",
                        f'![]({rel_image}){{width=13.333in height=7.5in}}',
                    ]
                )

        for row in measure_rows:
            summary_panel = Path(row["summary_panel"])
            if not summary_panel.exists():
                continue
            rel_image = os.path.relpath(summary_panel, slides_dir)
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    f'![]({rel_image}){{width=13.333in height=7.5in}}',
                ]
            )

    return "\n".join(lines) + "\n"


def main():
    ensure_dir(SLIDES_ROOT)

    manifests = [
        (STATS_ROOT / "power" / "manifest.csv", "Power"),
        (STATS_ROOT / "coherence" / "manifest.csv", "Coherence"),
    ]

    rows = []
    for manifest_path, measure_name in manifests:
        rows.extend(read_manifest(manifest_path, measure_name))

    rows = sorted(rows, key=sort_key)

    markdown_path = SLIDES_ROOT / "retrieval_mlmr_summary.md"
    pptx_path = SLIDES_ROOT / "retrieval_mlmr_summary.pptx"

    markdown_path.write_text(build_markdown(rows, SLIDES_ROOT), encoding="utf-8")

    subprocess.run(
        [
            "pandoc",
            markdown_path.name,
            "--slide-level=1",
            "-t",
            "pptx",
            "-o",
            pptx_path.name,
        ],
        check=True,
        cwd=str(SLIDES_ROOT),
    )

    print(f"Wrote {pptx_path}")


if __name__ == "__main__":
    main()
