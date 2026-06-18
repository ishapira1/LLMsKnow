# 00 Project Brief

## What this project is

This project studies sycophancy in large language models through a mechanistic interpretability lens. The core question is whether a model that gives a sycophantic answer is genuinely uncertain, or whether it internally supports the truthful answer and still follows the user’s false or misleading stance. We focus first on verifiable question answering settings, where correctness can be checked externally, and compare neutral prompts to bias-injected prompts that introduce user pressure.

## Motivation

Most work on sycophancy evaluates models behaviorally and treats them as black boxes. That makes it difficult to separate two different failure modes. In one case, the model is truly unsure and gets pushed into error. In the other, the model appears to carry internal evidence for the correct answer but still produces a compliant response. Recent work on hidden knowledge suggests that language models can sometimes know more than they show. This project asks whether sycophancy is often one of those cases. If so, sycophantic behavior may reflect a knowledge-expression gap rather than a simple competence failure. 

## What we are trying to achieve

Our goal is to distinguish uncertainty-driven sycophancy from policy-override sycophancy. We want to measure whether internal truth evidence remains stable or degrades when user pressure is introduced, how often sycophantic errors occur despite strong internal support for the correct answer, and whether these patterns differ across single-turn and multi-turn interaction regimes. More broadly, we want a mechanistic account of sycophancy that links external behavior to internal model state and helps clarify when the model fails because it does not know the truth versus when it knows the truth but does not say it. 

## Current working notes

The current near-term action items are tracked in [meeting_notes_2026-06-15.md](meeting_notes_2026-06-15.md). As of June 15, 2026, the priorities are to complete the full experiment pipeline after diagnosing cluster errors, rerun the pruning experiment using Hadas's code, develop the possible connection to in-context learning and mechanistic interpretability, and read the collected papers more carefully.

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
