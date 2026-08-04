# 00 Project Brief

## What this project is

This project studies sycophancy in large language models through a mechanistic interpretability lens. The core question is whether a model that gives a sycophantic answer is genuinely uncertain, or whether it internally supports the truthful answer and still follows the user’s false or misleading stance. We focus first on verifiable question answering settings, where correctness can be checked externally, and compare neutral prompts to bias-injected prompts that introduce user pressure.

## Motivation

Most work on sycophancy evaluates models behaviorally and treats them as black boxes. That makes it difficult to separate two different failure modes. In one case, the model is truly unsure and gets pushed into error. In the other, the model appears to carry internal evidence for the correct answer but still produces a compliant response. Recent work on hidden knowledge suggests that language models can sometimes know more than they show. This project asks whether sycophancy is often one of those cases. If so, sycophantic behavior may reflect a knowledge-expression gap rather than a simple competence failure. 

## What we are trying to achieve

Our goal is to distinguish uncertainty-driven sycophancy from policy-override sycophancy. We want to measure whether internal truth evidence remains stable or degrades when user pressure is introduced, how often sycophantic errors occur despite strong internal support for the correct answer, and whether these patterns differ across single-turn and multi-turn interaction regimes. More broadly, we want a mechanistic account of sycophancy that links external behavior to internal model state and helps clarify when the model fails because it does not know the truth versus when it knows the truth but does not say it. 

## Current working notes

The current empirical anchor is [current_empirical_summary_2026-06-20.md](current_empirical_summary_2026-06-20.md). The main experiment grid was rerun on June 20, 2026, and older result-number files in `sources/` have been retired or removed so future AI-assisted summaries do not reuse stale numbers.

The June 15 action items in [meeting_notes_2026-06-15.md](meeting_notes_2026-06-15.md) are now historical context. After the June 20 rerun, the priorities are to write from the updated empirical summary, keep probe claims calibrated as hidden-knowledge-like rather than causal proof, and plan intervention work such as patching or steering for stronger causal evidence.

The current near-term planning source is [meeting_notes_2026-07-24.md](meeting_notes_2026-07-24.md). The July 24 action items prioritize running the full weight-pruning experiment while staying as close as possible to the original code, sharing the exact flags and configuration with Hadas, diagnosing what failed in the mean-difference experiment, testing whether anti-sycophancy interventions also affect other behaviors such as in-context learning, and developing the paper write-up. The project is aiming for an ICLR submission.

### Latest weight-pruning conclusion (August 3, 2026)

The `diverse_templates` experiment indicates a **fixed-budget dilution effect**, not that broad behavioral diversity is intrinsically harmful. With 412 pruning examples in every condition, concentrating them on incorrect-suggestion adoption and doubt-induced errors outperformed dividing the same budget across 12 factual-pressure families: wrong-suggestion adoption was 14.0% versus 17.8%, and doubt-induced errors were 12.5% versus 23.4%. The 12-family condition accepted more valid corrections (71.0% versus 64.8%), so it was not uniformly worse. See the [final diverse-templates report](../artifacts/pruning/diverse_templates_remote/analysis/final_report.md).

The next clean diversity test should retain the complete original blocks and **add sufficiently many disjoint, behaviorally unique examples per new family**, rather than replacing or spreading the original examples. The pruning set should contain actual failures such as accepting an incorrect suggestion or abandoning a correct answer after doubt. The preservation set should include Alpaca/general capability data, rejection of incorrect suggestions, resistance to misleading doubt, stability under correct suggestions, and genuine corrections from an initially wrong answer. Diversity should mean unique questions and nonoverlapping prompt realizations with adequate coverage per behavior, not merely additional paraphrases of the same examples.

## Working definitions

**Sycophancy** is behavior where the model affirms a user’s stated or implied stance even when it conflicts with factual accuracy or sound judgment, instead of offering a direct correction or counterargument. 

**Hidden knowledge** is a case where the model’s internal representations support the correct answer more strongly than its final output suggests. 

**Knowledge-expression gap** is the gap between what the model appears to represent internally and what it ultimately says. 

**Bias injection** is the process of modifying a neutral prompt so that it includes a false, misleading, or strongly suggestive user stance. 

**Internal truth evidence** is any internal signal, such as a readout from intermediate activations, that indicates support for the correct answer.

**Uncertainty-driven sycophancy** is the regime where the model yields mainly because its internal evidence for the correct answer is weak, so user pressure tips an ambiguous decision. 

**Policy-override sycophancy** is the regime where the model yields even when its internal evidence for the correct answer is strong, which suggests that user alignment pressures override truthful correction at generation time. 

## Open problems

It is still unclear how reliably probe-based readouts capture genuine internal truth evidence rather than superficial correlates. We also do not yet know whether user pressure changes the model’s internal evidence itself, or mainly changes the final readout and decoding. Another open question is how common confident-yet-compliant failures are across models, tasks, and prompting regimes. It also remains open whether single-turn bias injections and multi-turn challenge prompts trigger the same underlying mechanism. Finally, even if hidden knowledge is present, we still need stronger causal evidence about which internal components drive the shift from truthful correction to sycophantic compliance.  
