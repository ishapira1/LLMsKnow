# Three Claims for Probe Interpretation in the Sycophancy Project

## What this note is for

This note gives a simple project-level framework for interpreting the probe results.

The main reason to use this framework is that the probe story is easy to overstate. A probe can work well in one setting, transfer partly to another, and still fail to be a true framing-agnostic truth detector. The three-claim framework helps separate those levels.

This note is meant to be used as a standing project reference, in the style of a ChatGPT Project context file.

## Project motivation

The broader project asks whether sycophancy is mostly:

1. **uncertainty-driven**, where the model yields because its internal support for the true answer is weak, or
2. **policy-override**, where the model still internally supports the truth but externally follows the user's false or misleading stance.

That question is hard to answer from behavior alone. A model can move toward the user's answer for at least two very different reasons:

- it may actually become internally less supportive of the truth
- it may still internally support the truth, but fail to express it

The probe program is meant to help separate these possibilities.

At the same time, the project brief already warns that probe-based readouts may capture genuine internal truth evidence, or they may capture superficial correlates. So probe results need to be interpreted in layers rather than all at once.

## The three claims

### Claim 1. Within-family decodability

A probe trained and tested on the **same prompt family** can recover correctness.

Examples:
- neutral-trained probe evaluated on neutral prompts
- incorrect-suggestion-trained probe evaluated on incorrect-suggestion prompts
- doubt-correct-trained probe evaluated on doubt-correct prompts

### Why this claim is attractive

This is the most basic and easiest claim.

If Claim 1 holds, then there is at least some usable correctness signal in the activations for that framing. It tells us the probe is not completely meaningless and that the model state contains information that can help distinguish correct from incorrect answers.

This is also the closest claim to the hidden-factual methodology, where the main goal is to rank correct answers above plausible wrong answers from internal activations.

### What Claim 1 does **not** mean

It does **not** mean the probe has found a framing-agnostic truth feature.

A within-family probe may still rely partly on framing-specific shortcuts. In particular, in stance-bearing prompts the probe may exploit properties of the user suggestion itself rather than a stable truth representation.

### What we think we have now

**Claim 1 is supported.**

That is one of the clearest results in the current package.

- matched-family probes perform well in their own framing
- truth is strongly decodable within neutral, incorrect-suggestion, suggest-correct, and doubt-correct families
- this means there is real correctness signal in the hidden state under each framing

So the current experiments clearly support within-family decodability.

---

### Claim 2. Same-readout stability

If we keep the **same neutral-trained probe fixed**, does it still work when the framing changes?

Examples:
- neutral-trained probe evaluated on incorrect-suggestion prompts
- neutral-trained probe evaluated on doubt-correct prompts
- neutral-trained probe evaluated on suggest-correct prompts

This is a cross-framing transfer claim.

### Why this claim is attractive

Claim 2 is more interesting than Claim 1 because it starts to test whether the correctness signal is shared across framings rather than relearned separately inside each framing.

If the same neutral probe still performs nontrivially after the prompt changes, then the readout is at least partly robust. That makes the probe more useful as a measurement device.

This claim also begins to separate two possibilities:

- the model stores truth in a reusable way across framings
- each framing needs its own specialized readout

### What Claim 2 does **not** mean

Even if the same probe still performs reasonably well after the framing changes, that does **not** mean the score is stable for the same candidate answer.

A probe can still pick the correct answer often enough to look useful while its score moves a lot with the user's stance.

So Claim 2 is a **transfer** claim, not an **invariance** claim.

### What we think we have now

**Claim 2 is supported only in a weaker sense.**

The current neutral-trained probe still has nontrivial cross-framing signal:

- it works well on neutral
- it degrades on incorrect-suggestion and doubt-correct
- it improves on suggest-correct
- it stays relatively stable on the congruent-suggestion control

So the same readout does not completely collapse under framing change. That is real.

But the drop is too large to call it a stable readout in a strong sense. The current evidence supports **partial cross-framing transfer**, not robust same-readout stability.

The congruent-suggestion control is especially useful here. It suggests that not all degradation is mere prompt distribution shift. When the new structure is aligned with the model's prior answer, the neutral probe behaves much more stably than it does under incorrect suggestion. That supports the idea that at least part of the drop is due to a real framing effect, not just out-of-distribution probe brittleness.

---

### Claim 3. Framing-agnostic truth detection

There exists a probe score that tracks **true versus false** while being largely insensitive to whether the user is:

- neutral
- suggestive
- doubtful
- misleading
- or otherwise changing the stance structure of the prompt

This is the strongest and most attractive claim.

It is the closest thing to saying that we have found a genuine **truth readout** rather than a family-specific decoder.

### Why this claim is attractive

Claim 3 is attractive because it is exactly the kind of result that would make the probe scientifically central to the project.

If Claim 3 held, then a strong policy-override interpretation would become much more plausible. We could say:

- the same truth-related score remains available across framings
- the model's surface behavior changes more than its internal truth readout does
- therefore at least some sycophantic failures look like a knowledge-expression gap rather than simple uncertainty

Claim 3 is also attractive because it would turn the probe into a much better instrument. It would mean the probe is not merely useful inside one framing family, but is tracking something closer to correctness itself.

### What Claim 3 would require

A strong claim-3 result would look like this:

- the same candidate answer gets roughly the same score when only the framing changes
- the score depends much more on correctness than on endorsement
- one shared readout works well across framings
- performance does not require retraining a different probe for each family

In other words, Claim 3 is an **invariance** claim, not just a **transfer** claim.

### What we think we have now

**Claim 3 is not established.**

This is the most important current conclusion.

The current results argue against a strong framing-agnostic truth detector:

- the same neutral probe changes substantially across framings
- the same candidate's score moves a lot under incorrect suggestion and doubt-correct
- the correct answer score often moves down when the user leans against it
- the endorsed wrong answer score often moves up when the user leans toward it

At the same time, the matched-family probes still do very well. That means truth remains **decodable within framing**, but it does not look like the same framing-agnostic readout survives unchanged.

The best current interpretation is:

- truth is still present in the activations to a meaningful extent
- but the representation or readout geometry shifts with framing
- so the current probes look more like **framing-conditioned truth readouts** than a universal truth detector

## How the three claims relate to each other

The three claims form a ladder.

- **Claim 1** asks whether correctness is decodable at all inside a prompt family.
- **Claim 2** asks whether the same readout still works somewhat when the framing changes.
- **Claim 3** asks whether that readout is largely invariant to framing.

So the hierarchy is:

- Claim 1 = within-family decodability
- Claim 2 = cross-family transfer of the same readout
- Claim 3 = framing-agnostic truth detection

A probe can satisfy Claim 1 without satisfying Claim 2.
A probe can satisfy Claim 2 in a weak sense without satisfying Claim 3.

That is where we are now.

## Current status of the project in this language

The current evidence supports the following picture:

### External side

The external story is already fairly strong.

- behavioral sycophancy is real and targeted toward the user-backed wrong answer
- the friction hypothesis is supported as a **movement** claim rather than only a flip claim
- weakly committed items move more toward the user-backed wrong answer under the same incorrect-suggestion intervention

This external result matters because it says sycophancy is not just a boundary-crossing artifact. It is a graded movement phenomenon.

### Internal side

The internal story is more nuanced.

- there is clear within-family decodability
- there is partial same-readout transfer
- there is not yet strong evidence for framing-agnostic truth detection

So the probe story is not empty, but it is also not yet the strong “truth machine” story.

## Best current interpretation

The most defensible interpretation right now is:

1. **Neutral hidden knowledge exists**, but it is partial and most visible on model-error items.
2. **The same neutral readout usually does not survive biased framing cleanly.**
3. **Truth remains strongly decodable within framing**, especially with matched-family probes.
4. **The best interpretation is representation shift or re-encoding**, not full erasure and not broad stable override.
5. **A small override-like slice may exist**, but it is not the dominant regime in the current results.

So the project should currently avoid saying:

> we have found a framing-agnostic truth detector inside the model

Instead, the safer statement is:

> we have evidence for within-family truth decodability and partial cross-framing transfer, but not yet for a robust framing-agnostic truth readout.

## Why this framework is useful for future conversations

This three-claim language is useful because it prevents common confusions:

- strong within-family performance does **not** imply framing-agnostic truth detection
- cross-framing performance drop does **not** by itself prove internal truth vanished
- a matched-family probe succeeding on biased prompts does **not** prove the same neutral truth readout survived

So future discussions should always specify:

- are we making a Claim 1 statement?
- a Claim 2 statement?
- or a Claim 3 statement?

That will make it much easier to keep probe claims calibrated.

## What would move the project toward Claim 3

The next experiments that matter most for Claim 3 are:

1. **Hadas's benign distribution-shift control**
   - neutral-trained probe on a new structure that should preserve labels and induce little real conflict
   - helps separate generic OOD brittleness from sycophancy-specific degradation

2. **Rephrasing robustness check**
   - same semantics, different wording
   - checks whether the probe is fragile to superficial variation

3. **Mixed-family probe training**
   - train on a balanced mix of framing families where the user suggestion is not predictive of correctness
   - tests whether a more framing-agnostic readout can be learned

4. **Leave-one-family-out evaluation**
   - train on several framing families and test on a held-out one
   - strongest non-causal benchmark for framing-agnostic truth detection

5. **More candidate-ranking analysis**
   - stay aligned with the hidden-factual methodology by emphasizing ranking metrics over plausible wrong answers, not only top-1 correctness

## Bottom line

The current project has a good three-level story:

- **Claim 1:** supported
- **Claim 2:** partially supported, but only in a weaker transfer sense
- **Claim 3:** attractive and important, but not established by current experiments

This is not a failure. It is actually a useful clarification.

It means the project has already learned something nontrivial:

- truth is decodable within framing
- framing changes the readout substantially
- the remaining open problem is whether there exists a shared truth readout that is robust to framing rather than merely relearnable inside each framing

That is the right target for the next stage of the project.
