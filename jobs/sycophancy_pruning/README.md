# Sycophancy Pruning Slurm Jobs

These jobs run the Hugging Face-only sycophancy weight-pruning experiment for `Qwen/Qwen2.5-7B-Instruct` on the Harvard cluster.

## Verification ladder

0. Local tiny CPU smoke:

   ```bash
   bash jobs/sycophancy_pruning/local_tiny_cpu_smoke.sh
   ```

1. Local tiny Qwen-family CPU smoke:

   ```bash
   bash jobs/sycophancy_pruning/local_qwen_tiny_cpu_smoke.sh
   ```

2. Cluster Qwen7B CUDA preflight:

   ```bash
   jobs/sycophancy_pruning/submit.sh preflight
   ```

3. Cluster smoke test:

   ```bash
   jobs/sycophancy_pruning/submit.sh smoke
   ```

4. Capped pilot:

   ```bash
   jobs/sycophancy_pruning/submit.sh pilot
   ```

5. Full two-dataset run:

   ```bash
   jobs/sycophancy_pruning/submit.sh full
   ```

Or submit the cluster stages as one dependency chain:

```bash
jobs/sycophancy_pruning/submit.sh chain
```

Optional single-dataset full runs:

```bash
jobs/sycophancy_pruning/submit.sh arc
jobs/sycophancy_pruning/submit.sh csqa
```

## Job files

- `preflight_qwen25_two_dataset.sbatch`: Qwen load plus strict-MC scoring on the known failing ARC row and a tiny per-dataset sample, no pruning.
- `smoke_qwen25_two_dataset.sbatch`: Qwen load plus tiny calibration/eval caps, sparsities `0,1e-5`.
- `pilot_qwen25_two_dataset.sbatch`: capped two-dataset run, sparsities `0,1e-6,1e-5,1e-4`.
- `qwen25_two_dataset.sbatch`: full two-dataset sweep, sparsities `0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3`.
- `full_arc_challenge_qwen25.sbatch`: full ARC-Challenge only.
- `full_commonsense_qa_qwen25.sbatch`: full CommonsenseQA only.
- `run_common.sh`: shared environment setup and command construction.
- `local_tiny_cpu_smoke.sh`: CPU end-to-end run using `HuggingFaceTB/SmolLM2-135M-Instruct`.
- `local_qwen_tiny_cpu_smoke.sh`: CPU end-to-end run using `Qwen/Qwen2.5-0.5B-Instruct`.

Every `.sbatch` job requests one GPU, `100G` memory, and sends `END,FAIL` email to `itaishapira@g.harvard.edu`.

## Environment requirements

The jobs expect:

- Repo at `/n/home12/ishapira/LLMsKnow`, unless `REPO_DIR` is set.
- Python at `/n/home12/ishapira/.conda/envs/itai_ml_env/bin/python`, unless `ENV_PYTHON` is set.
- `.env` in the repo root.
- `HUGGINGFACE_HUB_CACHE` or `HF_HUB_CACHE` set in `.env`, not under `/home`.

The shared runner also places temporary/cache-heavy paths in the same large-storage area:

- `HF_DATASETS_CACHE` defaults to `$HF_HUB_CACHE/datasets`.
- `HF_HOME` defaults to a sibling of `$HF_HUB_CACHE`.
- `TMPDIR` defaults to `$HF_HOME/tmp`; override with `SYCOPHANCY_TMPDIR` if needed.
- `MPLCONFIGDIR` and `TORCH_HOME` default under `$HF_HOME`.
- `OUT_DIR` defaults to `$(dirname "$HF_HUB_CACHE")/LLMsKnow_results/sycophancy_pruning`, unless `OUT_DIR` or `SYCOPHANCY_PRUNING_RESULTS_DIR` is set.

## Useful overrides

Slurm preserves exported variables, so you can override a run without editing the job:

```bash
RUN_NAME=my_pilot SPARSITIES_CSV=0,1e-5,1e-4 jobs/sycophancy_pruning/submit.sh pilot
```

To force a specific large results directory:

```bash
SYCOPHANCY_PRUNING_RESULTS_DIR=/n/holylabs/LABS/YOUR_LAB/Users/ishapira/sycophancy_pruning \
jobs/sycophancy_pruning/submit.sh pilot
```

For a larger pilot:

```bash
MAX_QUESTIONS_PER_DATASET=250 \
MAX_CALIBRATION_RECORDS=500 \
MAX_PRESERVATION_RECORDS=1000 \
MAX_EVAL_RECORDS=1500 \
jobs/sycophancy_pruning/submit.sh pilot
```

To pass extra CLI flags directly through to `run_sycophancy_pruning.py`:

```bash
jobs/sycophancy_pruning/submit.sh smoke --save_all_sweep_masks
```

To force an explicit dtype:

```bash
TORCH_DTYPE=bfloat16 jobs/sycophancy_pruning/submit.sh preflight
```

Outputs are written under the configured large-storage output root:

```text
<OUT_DIR>/Qwen_Qwen2_5_7B_Instruct/<run_name>/
```
