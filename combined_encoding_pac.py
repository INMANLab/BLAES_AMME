#!/usr/bin/env python
"""Combined PAC encoding analysis for the AMME+BLAES cohort only."""

import shutil
from pathlib import Path

from combined_pac_common import COMPOSITE_LOGIC_TEXT, augment_with_bla_composites, build_bla_composite_data, split_full_pac_data
from to_combine.BLAES_Group_PAC_analyses_from_matlab_encoding import (
    DATA_PATH,
    PAC_FILES,
    ensure_dir,
    export_summary_tables,
    load_blaes_encoding_pac,
    plot_bc_bar_by_memory,
    plot_bc_bar_graph,
    plot_bc_per_roi,
    plot_bc_remembered_forgotten,
    plot_connected_dots,
    plot_mi_difference_per_roi,
    plot_pac_by_patient,
    plot_pac_by_roi,
    plot_per_roi_patient_stim_nostim,
    plot_per_roi_quadrant,
    plot_quadrant_stim_memory,
    plot_stim_vs_nostim,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / 'outputs' / 'PAC_encoding'


def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_memory_data(data):
    memory_keys = [
        'post_stim_rem',
        'post_stim_forg',
        'post_nostim_rem',
        'post_nostim_forg',
        'diff_stim_rem',
        'diff_stim_forg',
        'diff_nostim_rem',
        'diff_nostim_forg',
    ]
    return any(data.get(key) for key in memory_keys)


def write_bla_composite_logic_note(out_dir):
    note_path = Path(out_dir) / 'BLAComposite_logic.txt'
    note_path.write_text(COMPOSITE_LOGIC_TEXT + '\n', encoding='utf-8')


def move_prefixed_outputs(src_dir, dest_dir, prefix='BLAComposite_'):
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        target = dest_dir / f'{prefix}{path.name}'
        if target.exists():
            target.unlink()
        path.replace(target)


def generate_common_plots(data, out_dir, label, footer_text=None):
    out_dir = ensure_dir(out_dir)
    print(f'  Generating common PAC plots for {label}...')
    plot_pac_by_roi(data, out_dir, label, footer_text=footer_text)
    plot_pac_by_patient(data, out_dir, label, footer_text=footer_text)
    plot_stim_vs_nostim(data, out_dir, label, footer_text=footer_text)
    plot_per_roi_patient_stim_nostim(data, out_dir, label, footer_text=footer_text)
    plot_bc_bar_graph(data, out_dir, label, footer_text=footer_text)
    plot_mi_difference_per_roi(data, out_dir, label, footer_text=footer_text)
    plot_bc_per_roi(data, out_dir, label, footer_text=footer_text)


def generate_memory_plots(data, out_dir, label, footer_text=None):
    if not has_memory_data(data):
        return
    out_dir = ensure_dir(out_dir)
    print(f'  Generating memory PAC plots for {label}...')
    plot_quadrant_stim_memory(data, out_dir, label, footer_text=footer_text)
    plot_per_roi_quadrant(data, out_dir, label, footer_text=footer_text)
    plot_bc_bar_by_memory(data, out_dir, label, footer_text=footer_text)
    plot_bc_remembered_forgotten(data, out_dir, label, footer_text=footer_text)
    plot_connected_dots(data, out_dir, label, footer_text=footer_text)
    plot_connected_dots(data, out_dir, label, footer_text=footer_text, show_patients=False)


def generate_bla_composite_outputs(data, out_dir, csv_dir, label):
    composite_data = build_bla_composite_data(data)
    has_any = any(composite_data.get(key) for key in ['post_all', 'post_stim', 'post_nostim', 'diff_stim', 'diff_nostim'])
    if not has_any:
        return

    print(f'  Generating BLA composite PAC plots for {label}...')
    write_bla_composite_logic_note(out_dir)

    tmp_fig_dir = reset_dir(Path(out_dir) / '_bla_composite_tmp')
    tmp_csv_dir = reset_dir(Path(csv_dir) / '_bla_composite_tmp')
    generate_common_plots(composite_data, tmp_fig_dir, f'{label} BLA Composite', footer_text=COMPOSITE_LOGIC_TEXT)
    generate_memory_plots(composite_data, tmp_fig_dir, f'{label} BLA Composite', footer_text=COMPOSITE_LOGIC_TEXT)
    export_summary_tables(composite_data, tmp_csv_dir)
    move_prefixed_outputs(tmp_fig_dir, out_dir)
    move_prefixed_outputs(tmp_csv_dir, csv_dir)
    shutil.rmtree(tmp_fig_dir, ignore_errors=True)
    shutil.rmtree(tmp_csv_dir, ignore_errors=True)


def load_grouped_encoding_pac_data():
    full_data = load_blaes_encoding_pac()
    return {
        key: augment_with_bla_composites(value)
        for key, value in split_full_pac_data(full_data).items()
    }


def main():
    print('=' * 60)
    print('Combined PAC Encoding Analysis')
    print('=' * 60)
    print(f'Data path: {DATA_PATH}')
    print(f'Output path: {OUTPUT_BASE / "all"}')
    print(f'Found {len(PAC_FILES)} PAC Phase 1 files.')

    grouped = load_grouped_encoding_pac_data()
    ensure_dir(OUTPUT_BASE)
    combined_data = grouped['all']
    combined_label = 'Combined AMME-BLAES'
    out_dir = reset_dir(OUTPUT_BASE / 'all')
    csv_dir = ensure_dir(out_dir / 'csvs')
    generate_common_plots(combined_data, out_dir, combined_label)
    generate_memory_plots(combined_data, out_dir, combined_label)
    export_summary_tables(combined_data, csv_dir)
    generate_bla_composite_outputs(combined_data, out_dir, csv_dir, combined_label)

    print('\n' + '=' * 60)
    print('Done! PAC encoding outputs saved to:', out_dir)
    print('=' * 60)


if __name__ == '__main__':
    main()
