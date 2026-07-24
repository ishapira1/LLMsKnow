# Vast weight-pruning micro-pilot

This bundle runs an explicitly preliminary, paper-faithful weight-space
micro-pilot on one Vast instance with two H100 GPUs.

Locked scope:

- Qwen2.5-7B-Instruct at the pinned revision.
- Calibration seed 5.
- Eight strict pruning rows and one eight-row mixed preservation manifest.
- Raw formatting and full-completion NLL.
- Transformer blocks 3, 8, 13, 18, 23, and 27 only.
- One targeted global set-difference mask at `p=1e-5`, `q=5e-5`.
- Live base-versus-pruned evaluation on 10 balanced held-out strict flips
  (five ARC and five CommonsenseQA), the largest balanced strict validation
  cohort available in the fresh 128-question-per-dataset sample.

The output label is:

> Preliminary six-layer, seed-5, N=8 weight-pruning micro-pilot — not the
> preregistered full experiment.

Persistent outputs and logs are written under
`/workspace/weight_pruning_micro_20260724`. Large temporary FP32 score files
are written under `/dev/shm/weight_pruning_micro_20260724`.

Install the supervisor configuration and start it with:

```bash
cp vast_supervisor.conf /etc/supervisor/conf.d/weight_pruning_micro.conf
supervisorctl reread
supervisorctl update
supervisorctl start weight_pruning_micro
```

Check progress with:

```bash
supervisorctl status weight_pruning_micro
cat /workspace/weight_pruning_micro_20260724/status.json
```
