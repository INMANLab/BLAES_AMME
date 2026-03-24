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


def band_display_name(band: str) -> str:
    mapping = {
        "theta": "Theta",
        "slow_gamma": "Slow Gamma",
        "fast_gamma": "HFA",
    }
    return mapping.get(band, band.replace("_", " ").title())


def level_display_name(level: str) -> str:
    return {
        "patient_level": "Patient Level",
        "trial_level": "Trial Level",
    }.get(level, level.replace("_", " ").title())


def sort_key(row):
    measure_order = {"Power": 0, "Coherence": 1}
    band_order = {"theta": 0, "slow_gamma": 1, "fast_gamma": 2}
    level_order = {"patient_level": 0, "trial_level": 1}
    return (
        measure_order.get(row["measure"], 99),
        band_order.get(row["band"], 99),
        level_order.get(row["level"], 99),
    )


def build_markdown(rows, slides_dir: Path) -> str:
    lines = [
        "---",
        "title: Retrieval Band-by-Region MLMR Summary",
        "---",
    ]

    for row in rows:
        overview_panel = Path(row["overview_panel"]) if row.get("overview_panel") else None
        if overview_panel and overview_panel.exists():
            rel_overview = os.path.relpath(overview_panel, slides_dir)
            lines.extend(
                [
                    "",
                    "---",
                    "",
                    f'![]({rel_overview}){{width=13.333in height=7.5in}}',
                ]
            )
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
        (STATS_ROOT / "power_band_by_region" / "manifest.csv", "Power"),
        (STATS_ROOT / "coherence_band_by_region_bla" / "manifest.csv", "Coherence"),
    ]

    rows = []
    for manifest_path, measure_name in manifests:
        rows.extend(read_manifest(manifest_path, measure_name))

    rows = sorted(rows, key=sort_key)

    markdown_path = SLIDES_ROOT / "retrieval_band_by_region_mlmr_summary.md"
    pptx_path = SLIDES_ROOT / "retrieval_band_by_region_mlmr_summary.pptx"

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
