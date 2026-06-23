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

### Current June 20 status

**Claim 1 is supported.**

That is one of the clearest results in the June 20 rerun.

- matched-family probes perform well in their own framing
- truth is strongly decodable within prompt families
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

### Current June 20 status

**Claim 2 is supported in a meaningful but incomplete sense.**

The June 20 rerun shows real cross-family transfer, but also a meaningful transfer penalty. The safest summary is:

- matched train/eval is clearly stronger than off-diagonal train/eval
- `random_all` is the best overall trained probe in the current grid
- random/doubt-style training families appear most framing-stable

So the readout does not collapse under framing change, but it is not perfectly invariant. The current evidence supports **substantial cross-framing transfer with a transfer penalty**, not a fully stable same-readout story.

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

### Current June 20 status

**Claim 3 is supported as hidden-knowledge-like evidence, but not as causal proof.**

This is the most important current calibration.

The June 20 rerun shows that, especially for `random_all`, a substantial correct-answer signal remains decodable even when the external model flips under strong wrong-user pressure. That is good evidence for a hidden-truth signal.

But this should not be overstated as a proven causal internal truth mechanism:

- cross-family transfer is real but imperfect
- paraphrases are mostly stable but not perfectly invariant
- activation movement is small-angle and structured, but the current package does not support every desired baseline
- intervention evidence such as patching or steering is still needed for the stronger causal claim

The best current interpretation is:

- truth remains decodable to a meaningful extent across biased framings
- `random_all` provides the clearest current hidden-knowledge-like evidence
- the project should still describe the result as evidence, not proof, of a hidden truth signal

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
- strong prompts make the external accuracy drop much larger
- high neutral confidence acts like friction against movement
- when errors happen, they are heavily targeted toward the user-backed wrong answer

This external result matters because it says sycophancy is not just random prompt sensitivity. It is targeted pressure toward the user's wrong answer.

### Internal side

The internal story is more nuanced.

- there is clear within-family decodability
- there is substantial cross-family transfer with a penalty
- `random_all` preserves a strong correct-answer signal even in many cases where the external model flips
- there is not yet causal proof that this signal drives or could control the final answer

So the probe story is strong enough to support hidden-knowledge-like evidence, but it is not yet an intervention-backed mechanism story.

## Best current interpretation

The most defensible interpretation right now is:

1. **External answers are highly sycophancy-sensitive**, especially under strong wrong-user pressure.
2. **Errors are targeted toward the user-backed wrong answer**, not randomly distributed.
3. **Neutral confidence acts like friction**, making confident correct answers harder to move.
4. **Truth remains substantially decodable**, especially through the `random_all` probe.
5. **The strongest causal claim still needs intervention evidence**, such as patching or steering.

So the project should currently avoid saying:

> we have found a framing-agnostic truth detector inside the model

Instead, the safer statement is:

> we have hidden-knowledge-like evidence that a correct-answer signal remains decodable across biased framings, especially with `random_all`, but the causal claim requires intervention work.

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

1. **Intervention work**
   - patching, steering, or related causal tests that ask whether the decodable truth signal can change the final answer

2. **More baselines for activation movement**
   - true random activation-pair cosine and same-family/different-question cosine if the artifacts support them in a future pull

3. **Stress-test the stable probe families**
   - especially `random_all`, `doubt_random_strong`, and `doubt_random`

4. **More candidate-ranking analysis**
   - stay aligned with the hidden-factual methodology by emphasizing ranking metrics over plausible wrong answers, not only top-1 correctness

## Bottom line

The current project has a good three-level story:

- **Claim 1:** supported
- **Claim 2:** supported with a meaningful transfer penalty
- **Claim 3:** supported as hidden-knowledge-like evidence, but not yet as causal proof

This is not a failure. It is actually a useful clarification.

It means the project has already learned something nontrivial:

- truth is decodable within framing
- framing changes the readout but does not erase it
- `random_all` is the strongest current probe family
- the remaining open problem is whether the decodable signal is causally usable for steering the model back to truth

That is the right target for the next stage of the project.
