# Python Environment
/Users/martinahollearn/anaconda3/envs/mne/bin/python

# Outputs
All outputs should go to outputs/ folder. Never write anywhere else

# Libraries
Never install libraries. Only use the existing python environment

# Version Control
After each time you update a script:

1. Run `git add .`
2. Run `git commit` with a useful message that clearly describes the change so the history is easy to follow and revert if needed

# Combined Notebook -> Python Workflow
When converting analysis notebooks in `to_combine/` into combined `.py` scripts:

1. Start from the original notebook logic, not from the existing combined script assumptions.
2. Compare each source notebook section-by-section against the combined `.py` loader and plotting logic.
3. Preserve notebook-specific filtering rules exactly, especially subject exclusions that are only valid for a specific region.
4. After editing the combined `.py`, run it with the Python environment above and verify it completes without errors.
5. Always regenerate notebook versions of the combined scripts after updating any combined `.py` file.
6. Treat the `.ipynb` regeneration as required work, not an optional cleanup step.

# Logic Checks To Double Check
Always verify these items when combining or updating scripts:

1. Subject exclusions must be applied only to the exact `Region + Patient` combinations used in the source notebooks.
2. Region cleanup rules must match the notebooks exactly, for example `ER -> EC`.
3. Trial/stimulation normalization must match the notebooks exactly, including which labels are dropped.
4. Memory-condition logic must match the notebooks exactly, including which rows are kept or dropped.
5. Frequency columns must be sorted numerically before averaging or plotting.
6. Any "overall" spectrum used in memory analyses must match the notebook logic exactly:
   either direct all-trial averaging or collapsing across available stim x memory condition spectra, depending on the notebook.
7. Baseline-corrected bar plots must use the same ROI exclusions as the notebooks for that dataset and analysis.
8. If BLAES stimulation is merged from another phase, confirm the merge keys and matched-row filtering match the notebook.
9. For combined `all` outputs, check that group-specific settings do not silently overwrite one another.

# Known Exclusions In This Project
Current notebook-matched subject+region exclusions:

- BLAES encoding: exclude `BJH042` and `BJH029` only when `Region == 'BLA'`
- AMME encoding: exclude `amyg016`, `amyg046`, `amyg057`, and `amyg037` only when `Region == 'BLA'`
- AMME retrieval: exclude `amyg057` only when `Region == 'BLA'`
- AMME retrieval: exclude `amyg030` only when `Region == 'HPC'`
- AMME retrieval: exclude `amyg034` only when `Region == 'CA'`

# Running Scripts
Use:

`env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=outputs/.mplconfig /Users/martinahollearn/anaconda3/envs/mne/bin/python <script>.py`

This keeps Python and Matplotlib cache writes inside `outputs/`.

# Python -> Notebook Conversion
After any update to a combined `.py` file, you must recreate the notebook version. To convert the combined scripts back into notebooks, run:

`/Users/martinahollearn/anaconda3/envs/mne/bin/python _convert_to_ipynb.py`

Notes:

1. `_convert_to_ipynb.py` preserves percent-format cells when present.
2. If a `.py` file is a plain script without `# %%` markers, it is converted into a notebook with one code cell containing the full script.
3. Do not stop after editing or running the `.py` file; the `.ipynb` must also be recreated before the task is considered complete.

# PAC Notebook Requirements
For the PAC notebooks in `to_combine/`:

1. The `.ipynb` file must remain runnable by itself from top to bottom without depending on an external wrapper script.
2. The notebook must define the paths and execution logic it needs inside the notebook-derived `.py` source.
3. When run as a notebook, figures must render inline in notebook output cells, not only save to disk.
4. After updating a PAC `.py` source in `to_combine/`, regenerate the matching `.ipynb` and verify the notebook-oriented plotting behavior is still preserved.
