#!/usr/bin/env python

import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
STATS_ROOT = OUTPUT_ROOT / "stats" / "ied_features"
SLIDES_ROOT = OUTPUT_ROOT / "slides"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_markdown(slides_dir: Path) -> str:
    lines = [
        "---",
        "title: IED Spatial/Structural Features Effects on Memory Modulation",
        "---",
    ]

    slide_images = []

    overview_dir = STATS_ROOT / "overview"
    for level in ("patient_level", "trial_level"):
        overview_path = overview_dir / f"{level}_overview.png"
        if overview_path.exists():
            slide_images.append(overview_path)

    # Patient-level outputs
    patient_dir = STATS_ROOT / "patient_level"
    for name in (
        "summary_panel.png",
        "coefficients_table.png",
        "model_comparison_table.png",
        "ChannelSpread_effect.png",
        "RegionSpread_effect.png",
        "Hemisphere_effect.png",
        "GrayMatterProp_effect.png",
    ):
        img = patient_dir / name
        if img.exists():
            slide_images.append(img)

    # Trial-level outputs
    trial_dir = STATS_ROOT / "trial_level"
    for name in (
        "summary_panel.png",
        "coefficients_table.png",
        "model_comparison_table.png",
        "random_intercepts.png",
        "ChannelSpread_effect.png",
        "RegionSpread_effect.png",
        "Hemisphere_effect.png",
        "GrayMatterProp_effect.png",
    ):
        img = trial_dir / name
        if img.exists():
            slide_images.append(img)

    for img_path in slide_images:
        rel_image = os.path.relpath(img_path, slides_dir)
        lines.extend(
            [
                "",
                "---",
                "",
                f"![]({rel_image}){{width=13.333in height=7.5in}}",
            ]
        )

    return "\n".join(lines) + "\n"


def main():
    ensure_dir(SLIDES_ROOT)

    markdown_path = SLIDES_ROOT / "ied_features_summary.md"
    pptx_path = SLIDES_ROOT / "ied_features_summary.pptx"

    md_content = build_markdown(SLIDES_ROOT)
    markdown_path.write_text(md_content, encoding="utf-8")

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
