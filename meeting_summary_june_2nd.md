**The big question**
- When the model goes along with a wrong answer, is it uncertain and has lost the truth, or does it still internally support the truth and only say the user's answer (override).
- The blocking sub-question right now is whether a moved probe reading reflects a real internal reorganization or just probe brittleness, and whether a probe change even means the model changed its mind.

**Where the analysis stands**
- External story is solid. Sycophancy is real and aimed at the wrong answer, friction is movement and not just flips, and at matched confidence what matters is how anchored the model is, not whether it was right.
- Internal story is mixed. Truth is decodable within a framing, the frozen neutral probe only partly carries over, and there is no framing-agnostic detector.
- The congruent control shows, behaviorally, that the damage comes from the disagreement and not from having a suggestion in the prompt. But this is one model and one dataset, and the internal congruent piece is newer.
- The activation mini-run, one run, gives three results:
  - Q1, displacement. The state moves a lot, about equally under incorrect and congruent, and more for the wrong answer than the gold. So the size of the movement is not conflict-specific, it is mostly the wrapper. This squares with congruent being behaviorally inert, lots of movement and little belief change.
  - Q2, parallel versus orthogonal. About ninety-nine percent of the movement is off the truth axis. The robust lesson is that the probe only sees the one-to-two percent parallel slice, so the accuracy change comes entirely from that small part. Whether that slice is above chance is still open, and the plot is not split by condition.
  - Q4, expected sign. Under incorrect, the wrong answer is reliably boosted, about eighty percent, but the correct answer is not reliably suppressed, about fifty-three percent. So the truth-axis effect is mainly pushing the wrong answer up, weak on pushing the correct one down.
- Q3, whether the truth direction itself rotates, is not done. It needs framing-specific probes on the matched subset, and it is the piece that would tell us whether the large orthogonal movement is re-encoded truth or just stance and noise.
- Read so far: a small, targeted truth-axis effect riding on a large off-axis shift, leaning against a simple lost-the-truth story, consistent with re-encoding rather than clean override, with the orthogonal mass as the open question.

**New things to work on**

*From the activation mini-run, immediate*
- Chance baseline for Q2. Compare the parallel fraction to a random direction of the same norm, and state whether the fraction is of the norm or the squared norm. This decides whether the bias targets or ignores the truth axis.
- Split Q2 by condition. Show parallel versus orthogonal separately for incorrect and congruent.
- Run Q3. Train a probe per framing on the matched items and compare its direction to the neutral probe against a split-half noise baseline, to see if there is a coherent rotated truth direction.

*Probe stability and a framing-agnostic readout*
- Rephrasing robustness check. Same questions, different wording, see whether the probe holds.
- Mixed-family probe. Train where the user suggestion carries no information about correctness.
- Leave-one-family-out. Train on several framings, test on a held-out one.
- Rebuild the full train-by-eval matrix, reporting top-1, pairwise ranking, and AUC.
- More candidate-ranking analysis, leaning on ranking over plausible wrong answers, not only top-1.
- Revisit the alternative probe constructions Hadas raised.
- Keep the congruent control as a standing sanity check, and inspect probe inputs and activations directly across neutral, congruent, and incorrect.
- Replicate the congruent and activation work beyond the one model and dataset.

*External and friction*
- Develop the friction result as a standalone writeup, centered on movement toward the wrong answer, its dependence on anchoring, and the matched-confidence c-versus-d comparison.
- Show the effect under more than one movement metric, raw probability plus KL or similar.
- Characterize the functional form of friction, whether it is linear, thresholded, or saturating.
- Compare weaker and stronger models to test whether competence changes susceptibility.
- Pin down the literature positioning for the friction claim.

*Mechanistic and conceptual follow-up*
- Pruning, on the hypothesis that sycophancy runs through a compact mechanism worth intervention-testing, building on Hadas's Nature work.
- Read "Sycophancy Is Not One Thing" and think through how it affects the design, since it argues sycophancy is not monolithic and that things like agreement and praise sit in separable directions, which bears on how we read the off-axis movement and how we build the probe.

*Discipline and bookkeeping*
- Keep internal claims modest until a genuinely stable probe turns up, and keep testing alternative readings where the model may still know the truth after bias.
- Avoid overclaiming that sycophancy is purely a capabilities problem, and stay cautious about reading robust cross-condition probe results as a clean truth representation.
- Keep organizing probe claims through the three-level framework rather than raw tables.
- Follow up on GPU access, and keep bookkeeping clean with the project summary updated after the rephrasing and mixed-probe checks.
