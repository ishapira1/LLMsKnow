When asked to generate code for a plot:
- Use `seaborn` and call `seaborn.set_style("white")`.
- Use large titles.
- Use axis-label font sizes of at least `15`.
- Use tick-label font sizes of at least `12`.
- Keep the color scheme consistent and predefined.
- If using two contrasting colors, use `#73b3ab` and `#d4651a`.
- Put the legend box below the plot.

When creating or editing Slurm batch scripts in this repo:
- Include `#SBATCH --mail-type=END,FAIL`.
- Include `#SBATCH --mail-user=itaishapira@g.harvard.edu`.
- Only omit or change those mail settings if the user explicitly asks.
- Prefer the sharded Slurm pattern for full sycophancy-bias experiments: sample once, then train/evaluate one probe family per job. Do not make a new monolithic all-dataset/all-family job unless the user explicitly asks.
- For the current full ARC/CSQA all-family/paraphrase experiment, use `jobs/sycophancy_bias_probe/full_allfamilies_paraphrase_sharded_20260616/` as the recommended reference bundle.
- Keep Slurm job names informative and short enough for scheduler display. Include the experiment family, stage, and date when practical, e.g. `syco_allfam_sample_20260616` or `syco_allfam_probe_20260616`.
- Organize logs in a structured tree under `jobs/sycophancy_bias_probe/logs/<bundle_name>/`, not as a loose flat pile.
- For new or substantially edited Slurm bundles, use this log layout:
  - `submit/` for submitter dry-run and submission logs.
  - `slurm/<stage>/` for raw Slurm stdout/stderr, such as `slurm/sampling/` and `slurm/probes/`.
  - `by_task/<dataset_model>/<stage_or_probe_family>/job_<job_id>/task_<array_task>.out` for canonical browseable task logs.
  - Matching `.err` paths beside `.out` files.
- Each task log should print enough startup/shutdown metadata to diagnose a failure without relying on the Slurm email: task label, model, dataset, probe family when applicable, run name, run directory, command, Slurm IDs, hostname, working directory, start/end time, exit status, elapsed seconds, and resource snapshots such as `nvidia-smi` / `sstat` when available.
- Keep operational cleanup explicit. Do not remove `.run.lock` files by default; require an explicit opt-in such as `ALLOW_STALE_LOCK_CLEANUP=1`.
- Keep conservative defaults for heavy Hugging Face Slurm jobs unless there is a clear reason to change them: `SAMPLE_BATCH_SIZE=1`, `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`, `MALLOC_ARENA_MAX=2`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- When adding or editing Slurm scripts, run `bash -n` on the changed `.sbatch` and submit scripts, and run `DRY_RUN=1` for submit wrappers when available.
- For more detailed sycophancy-bias Slurm conventions and current bundles, read `jobs/sycophancy_bias_probe/README.md`.
