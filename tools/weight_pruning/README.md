# In-repository weight-pruning engine

This directory contains the paper-faithful pruning runtime used by
`jobs/sycophancy_pruning/paper_global_sharded_20260722/`.

The seven runtime modules were imported from `orgadhadas/harm_pruning_WIP`
commit
`651e81657162df8f208b3acbd4638bb2f3a7f543`. They are kept inside LLMsKnow so
the experiment has one repository, one Git revision, and one Harvard checkout.
The Slurm bundle invokes `tools/weight_pruning/prune.py` directly; no external
pruning checkout, submodule, or symlink is used.

The in-repository copy adds postponed annotation evaluation for Python 3.9
import compatibility and pins the WikiText evaluation dataset revision. All
scientific pruning and evaluation logic otherwise retains the imported
implementation.
