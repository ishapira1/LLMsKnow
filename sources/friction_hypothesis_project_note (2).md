# Friction Hypothesis

## Claim

The **friction hypothesis** says that a model is harder to move when it is already more strongly committed to the true answer.

In this setting:

- `c` is the true answer
- `b` is the user-backed wrong answer

The weak version of the claim would be:

- low-confidence items flip more often

That version is not enough, because if an item starts with a larger gap between `c` and `b`, then it is mechanically harder to flip even under the same pressure.

The stronger and correct version is:

- under the same incorrect-suggestion intervention, items with **weaker neutral commitment** move **more toward `b`**

So friction is a **movement** claim, not just a **flip** claim.

## Intuition

We do not want to say only that weak items cross the decision boundary more often.

We want to say that when the model starts out less committed to the true answer, the incorrect-suggestion prompt changes its distribution more strongly in the direction of the user-backed wrong answer.

That means the right things to measure are quantities like:

- how much `P(b)` increases after the biased prompt
- how much the neutral gap `P(c) - P(b)` shrinks after the biased prompt

The cleanest version is even stronger:

- weakly committed items move more toward `b` **even when they do not flip**
- and even when they **stay correct**

## Bottom lines

- **Flip rate alone is not enough.**  
  A higher flip rate for weak items does not by itself prove friction, because weak items are closer to the decision boundary.

- **The right object is movement, not boundary crossing.**  
  The real test is whether the biased prompt causes a larger shift toward `b`, not just a larger number of flips.

- **The data support friction as a movement phenomenon.**  
  Under the same incorrect-suggestion prompt, weakly committed items move much more toward the user-backed wrong answer than strongly committed items do.

- **This remains true after removing actual sycophantic flips.**  
  Even among items that never flip to `b`, weakly committed items still shift more toward `b`.

- **This remains true even on items that stay correct.**  
  So the effect is not just “weak items flip more.” It is visible even when the model’s final answer does not change.

- **The result is robust to how commitment is defined.**  
  It appears when neutral commitment is defined using:
  - `P(c) - P(b)`
  - `P(c) - P(best wrong)`
  - the neutral probe margin `S(c) - S(b)`

- **The result appears in every run separately.**  
  It is not being driven by only one model or one dataset.

- **What the data do support:**  
  Weak neutral commitment predicts larger movement toward `b` under the same intervention.

- **What the data do not support:**  
  We cannot say that every item receives the exact same latent additive push, and weak items only flip more because they are closer to threshold.

- **Paper-safe formulation:**  
  Friction is real as a **distributional movement** phenomenon, not just as a **flip** phenomenon.

- **One-sentence summary:**  
  Weakly committed items are not just easier to flip, they are genuinely easier to move.

## Suggested use in Project context

This note is intended to be kept as a compact reference inside the Project so that future conversations can reuse the same definition of friction and avoid drifting back to the weaker flip-based version.
