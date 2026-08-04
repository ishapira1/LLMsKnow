# Dense behavioral-diversity pruning (`2026-08-03`)

This campaign tests whether broad factual-pressure pruning can produce a
Pareto improvement when diversity is **added densely** rather than obtained by
splitting a fixed 412-row budget.

The frozen Narrow-1x blocks remain present. A new 4,096-question candidate
pool is sampled under twelve factual-pressure families. Every family supplies
256 unique-question failures to pruning and 256 unique-question successful
resistance examples to preservation. Preservation additionally contains 512
neutral-correct examples, 512 correct-suggestion stability examples, 512
genuine neutral-wrong-to-correct updates, and 2,060 total Alpaca examples.

Scores average within each behavioral family before combining family means.
Four profiles vary only the weight on the central suggestion/doubt families
and on useful correction-taking. Each profile receives the same nine-point
`q × p/q` sweep plus a matched-996 sensitivity mask. Screening freezes three
global Pareto representatives for the full factual, paraphrase, utility,
MMLU, symbolic-ICL, SycoBench, and ELEPHANT evaluations. SycophancyEval
feedback judging is excluded under the user's standing opt-in-only decision.

Hypotheses:

1. Dense per-family support transfers better than the previous diluted
   twelve-family arm.
2. Explicit, dense successful-resistance and genuine-update preservation
   prevents the prior stubbornness tradeoff.
3. At least one profile reduces wrong-suggestion adoption and doubt-induced
   errors while keeping valid updates, neutral accuracy, and general
   capabilities near the base model.

Validation and submission:

```bash
bash -n common.sh cpu_stage.sbatch gpu_stage.sbatch submit.sh
python -m unittest -v test_campaign.py
DRY_RUN=1 ./submit.sh
DRY_RUN=0 ./submit.sh
```

No stage deletes `.run.lock`. Every Slurm script retains Harvard `END,FAIL`
mail, and custom progress/final emails are receipt-deduplicated.
