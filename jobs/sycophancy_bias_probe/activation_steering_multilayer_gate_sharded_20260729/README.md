# Simultaneous multilayer conditioned-steering gate

This bounded retry applies the saved ARC belief-conflict direction at every
nonterminal residual layer in the same forward pass. It reuses the frozen
conditioned-direction artifacts and neutral-correct ARC validation cohorts from
the completed 2026-07-26 experiment.

The primary all-layer dose is aggregate-energy controlled:

```text
per-layer normalized ratio = aggregate ratio / sqrt(number of layers)
```

Thus the root-sum-square of normalized layer doses equals the requested
aggregate ratio. The previously selected single layer is rerun in the same
scientific shard as the comparator. Both boundary-only and energy-matched
prompt-suffix positions are evaluated.

Run `bf16` first. Continue to `validation` only after both array tasks have an
exact all-layer alpha-zero no-op and no nonfinite values. Run `selection` only
after both validation tasks complete. No broader DAG is submitted automatically.
