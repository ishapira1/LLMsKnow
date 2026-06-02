# Meeting Record: April 7, 2026

## Meta information

- **Date:** 2026-04-07
- **Project:** R1: Sycophancy mechanistic interpretability
- **Participants:** Itai, Hadas
- **Meeting language:** Hebrew
- **Prepared as:** cleaned English transcript for bookkeeping
- **Source material:** noisy Hebrew transcript, prior meeting notes from 2026-04-02, and the surrounding project context and results discussed in this conversation
- **Speaker mapping used in reconstruction:**
  - **Speaker 0 = Itai**
  - **Speaker 1 = Hadas**
- **Important note:** this is a reconstructed and cleaned transcript, not a literal line-by-line translation. Obvious ASR errors, duplicated fragments, and malformed phrases were corrected using the project context. Where exact wording was unclear, the reconstruction preserves the most likely intended meaning.

## Main topics discussed

1. Probe cross-condition generalization and what the current probe table does and does not show.
2. External results on movement toward the user-backed wrong answer under incorrect suggestion.
3. The friction hypothesis as a movement phenomenon rather than only a flip phenomenon.
4. A new check comparing movement toward \(b\) when the model is confident in the correct answer \(c\) versus when it is confident in another answer \(d\).
5. A simplex-style visualization of output-distribution movement.
6. Concerns about probe stability and a proposed rephrasing robustness check.
7. A proposal to train a framing-agnostic probe by mixing prompt families so the user suggestion is not predictive of correctness.
8. Hadas's Nature-submitted pruning work and possible conceptual links to sycophancy.
9. Near-term action items and research priorities.
10. A brief unrelated discussion about Israel and travel.

---

## Executive summary

The meeting centered on two parallel threads.

The first was the **probe story**. The current probe results remain mixed. Neutral-trained probes degrade out of distribution, and probes trained on incorrect suggestion appear to rely partly on the bias structure itself. The congruent-suggestion control looks much better than incorrect suggestion, but strong cross-framing generalization is still not there. The group agreed that the right next diagnostic is a **rephrasing robustness check**, followed by a more serious attempt at a **framing-agnostic mixed training distribution** where the user suggestion is uninformative about truth.

The second was the **external story**, which currently looks stronger. The model’s output distribution clearly moves toward the user-backed wrong answer \(b\) under incorrect suggestion. The key friction question is how this movement depends on neutral commitment. A new check compared two cases with the same confidence level: one where the model is confident in the correct answer \(c\), and one where it is equally confident in some third answer \(d\). The discussed interpretation was that the movement toward \(b\) looks similar in those two cases, suggesting that **confidence or self-commitment matters more than correctness itself**.

The meeting also sharpened a conceptual point: sycophancy should be analyzed primarily as a **distributional movement phenomenon**, not only through discrete flips. The group discussed whether the observed friction pattern could partly depend on the metric used to measure movement, and agreed that the result should be shown under more than one movement metric.

Finally, Hadas described her new pruning work, submitted to *Nature*, which localizes harmful-generation behavior to a small set of weights. This raised the possibility that sycophancy might also rely on a relatively compact generation mechanism that could, in principle, be studied by targeted interventions such as pruning.

---

## Full reconstructed transcript

### 1. Probe table, cross-condition generalization, and what the current probe results do and do not show

**Itai:**  
Let me start with the probe side. The probe trained on incorrect suggestion is interesting, but I’m not sure yet whether the story is good news or bad news.

What I’m looking at is the table where I train on one condition and evaluate on another, and I look at top-1, the ranking metric from the hidden-factual framework, and AUC. In neutral, when I evaluate on neutral, performance is good. When I move out of distribution, performance drops. So the question is whether there is any real stability there, or whether this is all too fragile.

Then I added the congruent-suggestion condition, meaning: instead of suggesting a wrong answer, I suggest the answer the model would already have given. That should be a cleaner out-of-distribution sanity check. It changes the prompt format, but it should not create the same kind of conflict as an incorrect suggestion.

And there, performance looks better. The drop is very small. It is much closer to neutral-on-neutral than the incorrect-suggestion case is. That is what I was hoping to see.

**Hadas:**  
Right. So the drop there is much smaller.

**Itai:**  
Yes. Almost no drop. Around five percentage points or so, depending on the exact metric. So that looks fairly stable.

But then on suggest-correct and doubt-correct, performance is terrible. That part is really not good.

**Hadas:**  
And that is interesting, because it suggests the probe may be using the bias structure itself.

**Itai:**  
Exactly. That is what worries me. If the incorrect-suggestion-trained probe collapses on suggest-correct and doubt-correct, then maybe it learned something very tied to that specific framing. Maybe it is not learning “truth versus falsehood” in a framing-agnostic sense. Maybe it is learning something much more conditional.

**Hadas:**  
Which metric do you think is the most meaningful here?

**Itai:**  
Probably AUC, at least if the question is just whether it distinguishes correct from incorrect. Because with incorrect suggestion there is already a cue in the prompt. The suggestion itself is informative in that setting. Then if I flip the structure and use suggest-correct, it makes sense that performance may deteriorate, because the cue it learned is now misleading.

**Hadas:**  
Yes. If the user-suggested answer is sometimes the correct one and sometimes not, that breaks the shortcut.

**Itai:**  
Right. Which means the probe may really be exploiting that bias cue more than I expected.

**Hadas:**  
More than I would have expected too. Especially because only a fraction of the points should directly match the suggestion token itself.

**Itai:**  
Yes, exactly. So the size of the drop still feels larger than I would have guessed from that alone.

That’s why I feel the whole evaluation story here is still unstable. It’s not that there is nothing to say. There is something to say. But it’s still shaky ground. I don’t yet feel comfortable making a strong claim from these probe numbers alone.

**Hadas:**  
That seems right.

**Itai:**  
One concrete sanity check I want to run next is rephrasing. Just take the same question and ask another model to rewrite it in different words, with no meaningful semantic change. Then evaluate the same probe on those rephrasings. If performance drops a lot there too, then that tells us something very important. It means the probe is fragile even to superficial variation.

**Hadas:**  
That is a very good check. Because in principle it should generalize across paraphrases. If it doesn’t, then maybe it is just tied to a narrow distribution of phrasing.

**Itai:**  
Yes. That is exactly what I’m worried about. Maybe the whole thing is partly an artifact of a particular distribution of question wording.

So at the moment, internally, I still feel stuck on the same core issue: I do not yet have a probe I trust as a stable measurement device.

---

### 2. External results look clearer than internal ones

**Itai:**  
Externally, though, things look more solid. The internal side is still messy because the measurement question is messy. But the external side is clearer.

What we know externally is that once I inject an incorrect suggestion, probability mass shifts toward the user-backed answer \(b\). The whole distribution moves in that direction.

That part is already clear.

Then the friction question is: how does that movement depend on confidence?

And the new thing I wanted to test was this: suppose I match on confidence. In one case the model is confident in the correct answer \(c\). In another case it is equally confident, but in some third answer \(d\), which is neither correct nor the user-suggested answer. If I then inject the same incorrect suggestion toward \(b\), do I get the same movement toward \(b\), or not?

**Hadas:**  
Right. Because if it is really just a confidence story, then those two cases should behave similarly.

**Itai:**  
Exactly. And what I’m seeing is that they do behave very similarly. What really seems to matter is the confidence itself, not whether the model’s original belief was correct.

That is why I made that new plot.

---

### 3. The simplex plot and the “same confidence, \(c\) versus \(d\)” result

**Itai:**  
I made a simplex plot. Each point is a probability distribution over three coordinates:
- probability on the correct answer
- probability on the user-backed wrong answer \(b\)
- probability on all the other answers combined

So each point is one item’s distribution before or after the intervention. Then I plot arrows for the movement.

I also do some smoothing so it’s visually readable.

**Hadas:**  
Okay. So bottom left is the combined “other” mass?

**Itai:**  
Yes. “Other” is just the sum of the remaining answers.

**Hadas:**  
And what do we see?

**Itai:**  
First, unsurprisingly, we see that things move toward \(b\). We already knew that.

The more interesting point is the one I just mentioned: when I compare cases with the same confidence level, it really does not seem to matter much whether the model initially believed \(c\) or believed some third answer \(d\). The displacement toward \(b\) looks roughly the same.

**Hadas:**  
So your current read is that what matters is confidence, not correctness.

**Itai:**  
Yes. That is my current read. I also checked this more directly with bucketed analyses, not only with the simplex plot. I bucket by confidence and compare the shift toward \(b\) using both margins and raw probabilities. And I keep getting basically the same story.

So I do not think the exact identity of the model’s original answer is adding much once confidence is controlled for.

**Hadas:**  
That strengthens the external conclusion quite a bit. It suggests the external signal is really about how anchored the model is, not about whether it is objectively right.

**Itai:**  
Yes. That is exactly how it feels to me. The output behavior seems to be telling us a lot about susceptibility to sycophancy. Not because the model is right or wrong per se, but because it is more or less anchored.

---

### 4. Movement matters more than flips

**Itai:**  
Another thing that feels important here is the measurement target itself.

A lot of papers measure sycophancy by asking whether the answer flipped. I really do not think that is the right object. The right object is how much the distribution moved.

If something starts 50-50, then a small push can create a flip. But if something starts 90-10, you may get no flip at all and still have a very real movement. If you only look at flips, you miss that.

So I think the correct definition is: how much does the prompt move the model toward \(b\), not just how often it crosses a boundary.

**Hadas:**  
Yes, though the movement should still be smaller when the model is more confident.

**Itai:**  
Right. That is the friction story. The movement is smaller when the model is more confident. I do not know of prior work that states it exactly that way. I looked for it and did not really find it.

So the story becomes:

1. sycophancy is a distributional movement phenomenon, not only a flip phenomenon  
2. the magnitude of movement depends on how anchored the model is  
3. that anchoring seems to depend mostly on confidence, not specifically on correctness

**Hadas:**  
That sounds like a good external claim, but it still needs to be shown carefully.

**Itai:**  
Yes. I agree. I think the evidence is there, but it still needs to be tightened.

---

### 5. A caution about the metric itself

**Itai:**  
There is also an important caveat.

All of this friction analysis so far depends on how we measure movement. Right now I’m mostly plotting raw probability movement. But that might not be the only or even the best metric.

For example, maybe what really matters is KL divergence or some other distributional distance. If every item moved by the same amount in KL, the raw probability-space plots might still make it look like low-confidence items moved more, just because the geometry of probability space is not uniform.

So one question is whether our friction story is partly a metric artifact.

**Hadas:**  
Right. A different metric could tell a different story.

**Itai:**  
Yes. I do not think that is what is happening, because I checked KL too. But I still think it is important to show the effect under more than one metric.

**Hadas:**  
I agree. That always makes a result more convincing. Also, KL is not easy to interpret locally. It tells you distributions are different, but not in a way that is always easy to map back to a concrete behavioral change.

**Itai:**  
Exactly. That is why I do not want to rely on KL alone either. One metric can easily hide what is really happening. So I think we should show the result under multiple movement metrics.

I also tried to think about a theoretical model, something like: when bias is injected, the model solves some local optimization problem, staying close to its original distribution while moving toward the user-backed answer. I played with a few versions of that, but I do not yet have a satisfying theory that reproduces the empirical patterns cleanly.

So for now, I think the right order is: first get the empirical story really clear, then try to build the theoretical frame around it.

**Hadas:**  
That makes sense.

---

### 6. Probe interpretation, hidden knowledge, and what would count as stronger evidence

**Hadas:**  
Let me go back to the probe for a second, because I had a thought.

Suppose you manage to train a probe that works well across all the settings, including rephrasings and all the prompt framings. Then I think you can say something meaningful: namely, that the internal representation contains the truth information. The model still knows.

But the converse is not true. If you fail to find such a probe, that does not prove that the information is not there.

**Itai:**  
Yes. Exactly. Existence is informative. Non-existence is not.

If we did find a stable probe like that, it would be very strong evidence for the “internal truth remains intact” story. That would support the picture where the model still knows the truth but sometimes externally goes along with the user.

But right now I do not think we are there.

What worries me is not that hidden knowledge disappears completely after bias. It is that the internal representation itself seems to change in a meaningful way once I add the bias. So even though there is still some hidden knowledge, it is not obviously the same hidden knowledge.

That is why I do not trust the strong claim that “the model still knows the truth and simply chooses not to say it.” I do not think we have evidence for that yet.

**Hadas:**  
Yes. The model may actually be getting convinced by the user.

**Itai:**  
Right. That is the alternative story. There are really two pictures:

One picture is: internally the model still knows the truth, but externally it chooses the user-pleasing answer.

The other picture is: once the user confidently suggests something, the model internally shifts too. It actually becomes less sure of the truth or even believes the wrong answer more.

A stable probe that works everywhere would be strong evidence for the first picture. Failure to find one is not definitive, but it is at least suggestive of the second picture.

**Hadas:**  
Yes. That feels right.

---

### 7. A better probe training distribution

**Hadas:**  
One concrete thing I would try is changing the training distribution for the probe.

Right now the problem is that the suggestion itself may be informative. So instead of letting the suggestion correlate with truth or falsehood, train on a mixed distribution where the suggestion is explicitly not informative.

For example: sometimes the user says “I think the answer is X” and X is correct. Sometimes X is incorrect. Likewise for the doubt format. Mix them all together so that the probe has to ignore the user suggestion as a cue.

**Itai:**  
Yes. So basically train on the union of all the bias formats, with the property that what the user suggests carries no reliable information about correctness.

**Hadas:**  
Exactly.

**Itai:**  
That makes sense. We can create incorrect suggestion, suggest correct, doubt correct, and the corresponding opposite cases, and mix them so that the user framing itself is not predictive.

**Hadas:**  
Yes. Then if the probe still works well, that would be much more interesting.

**Itai:**  
Agreed. That seems like the right one more serious push before giving up on the probe story.

---

### 8. Pruning as an interpretability tool, and the Nature paper

**Hadas:**  
This also makes me think of pruning. Maybe pruning can help here.

**Itai:**  
What do you mean exactly?

**Hadas:**  
I mean pruning as an interpretability tool, not compression in the usual engineering sense.

I have a paper we just submitted to *Nature*. It is not public yet. The main result is that you can localize harmful-generation behavior to a very small set of weights. If you prune those weights, the model largely loses the ability to generate harmful content, even under fairly strong jailbreaks, while keeping most of its general capabilities.

Importantly, the model does not forget the concepts themselves. It can relearn the harmful behavior with a bit of fine-tuning. So what we are removing is not the underlying world knowledge. We are removing a generation mechanism.

**Itai:**  
So the mechanism is about producing the behavior, not about storing the knowledge.

**Hadas:**  
Exactly. And the really striking thing is that the mechanism is very compact. It is a tiny subset of weights.

We think this happens because alignment training compresses a broad family of refusal-related behaviors into a compact mechanism. Then if you prune that mechanism, you can selectively affect the behavior.

We also see related effects for misalignment. Narrow fine-tuning on some bad behavior can spill over into broader bad behavior, and pruning those weights can reduce that spillover too.

**Itai:**  
That is really interesting.

So the analogy here would be: maybe sycophancy is also mediated by a relatively compact mechanism that reads the user’s preference and steers the model in that direction, regardless of truth.

**Hadas:**  
Yes. I would not guarantee that. Not every behavior is necessarily localizable that way. But sycophancy seems like the kind of thing that might be, because it is a fairly triggerable behavior and it is closely tied to RLHF and instruction tuning.

**Itai:**  
And if such a mechanism exists, pruning it might reduce the tendency to move toward what the user wants to hear.

**Hadas:**  
Possibly. Though it could also damage other useful behavior. That is part of what would be interesting to test.

**Itai:**  
Right. Because some amount of understanding what the user wants is actually useful. You just do not want it to override truth.

**Hadas:**  
Exactly.

---

### 9. Details of the pruning paper

**Itai:**  
How do you identify which weights to prune?

**Hadas:**  
The algorithm is simple. You collect generations that represent the behavior you want to remove. In our case, harmful generations, often from a jailbroken version of the same model or a very nearby variant. Then you define a loss on those generations and estimate, for each weight, what would happen to that loss if you zeroed the weight out. It is basically a first-order Taylor approximation. That gives you a score per weight. Then you rank weights and prune the ones that most strongly support the bad behavior, while controlling against utility loss using a utility dataset.

**Itai:**  
So you compute something like a gradient-based saliency score for each weight, multiplied by the weight value itself.

**Hadas:**  
Yes, essentially.

**Itai:**  
And the generations come from a nearby model so that the loss is still meaningful in-distribution.

**Hadas:**  
Exactly. That matters a lot.

**Itai:**  
Do you think this could generalize across models?

**Hadas:**  
Not directly in the sense of “find weights on one model and prune another.” But as an algorithmic approach, yes. It is straightforward to run per model.

**Itai:**  
That makes sense.

---

### 10. How pruning might connect to sycophancy

**Itai:**  
I can imagine a hypothesis here.

Suppose sycophancy really is partly implemented by a compact set of weights that reward matching the user’s preferred answer. Then pruning those weights might make the model more truth-anchored.

In the kind of simplex plot I showed, maybe after pruning the arrows would become smaller, or maybe the movement toward \(b\) would be reduced.

**Hadas:**  
Yes, though it could also create other pathologies. Maybe the model becomes less helpful or less responsive in other ways.

**Itai:**  
Sure. It might damage normal user modeling too.

Still, it feels very relevant as a possible mechanistic follow-up.

**Hadas:**  
I agree. I would treat it as an additional direction, not a replacement for squeezing more out of the probe story first.

---

### 11. Where to focus next

**Itai:**  
So strategically, I think we are converging on the following:

On the probe side, we give it one more serious push:
- train on a broader mixed distribution of prompt framings
- test rephrasing robustness
- keep checking whether we can get something genuinely stable

On the external side, keep pushing the confidence and friction story, because the evidence there already looks meaningful.

And in parallel, maybe start thinking about pruning-based analysis as a more direct mechanistic intervention.

**Hadas:**  
Yes. I think that is the right plan.

I would definitely not stop now before doing the rephrasing check. Even on the neutral probe, that seems like an important diagnostic.

And I do think the external result is worth continuing. It already gives you a reasonably strong story, even if the internal story remains incomplete.

**Itai:**  
Yes. I agree. No matter what happens with the probe, the external story seems worth writing up as its own thing.

---

### 12. GPUs, code, and implementation logistics

**Hadas:**  
Did anyone respond to your GPU access request?

**Itai:**  
No. Not yet. I submitted the request, emailed the Harvard cluster people, and opened a ticket. No answer.

At this point the issue is mostly annoying rather than blocking. I already generated most of the responses I needed and saved them. So I do not need heavy generation nearly as much anymore. For the probe work I mostly need to load models and extract activations. That is still easier with GPUs, but it is less painful than generation.

**Hadas:**  
Yes. Training and pruning will still need GPUs because of backpropagation.

**Itai:**  
Right. So I still need them. But for now it is manageable.

**Hadas:**  
And on the pruning paper side, I’m cleaning the code now. I can share it before it is polished, but it is not yet at a public-release standard. Cleaning it properly will take time, and right now my priority is getting out the arXiv version.

**Itai:**  
That’s fine. Even rough code would still be useful for me once the preprint is out.

**Hadas:**  
Yes, that is doable.

We also talked a bit about whether LLM tools can help clean code, and my view is that they can help some, but not in a way that fully removes the need for careful manual review. Especially if I want the released code to be clean and readable enough that people will actually use it.

**Itai:**  
That makes sense.

---

### 13. A meta-point about the research process

**Itai:**  
One thing that’s been frustrating for me is that this project feels much more like detective work than theory-driven confirmation.

In some of my earlier projects, we had a strong theoretical picture first. Then the empirical work was mostly checking whether the expected pattern appears, or whether there is a bug.

Here it’s different. I keep asking a question, running a targeted analysis, getting a partial answer, and then that creates two more questions.

So progress feels spiral-shaped. Not linear.

**Hadas:**  
I actually think you are making good progress. It just feels less linear because the structure of the project is different.

**Itai:**  
Yes. That is probably right.

---

### 14. Brief unrelated Israel discussion

**Hadas:**  
Then we shifted to Israel and Brazil travel.

I said I was probably flying close to the conference, around the 20th, and we talked about whether the general situation might affect people traveling, whether the university might approve or disallow certain plans, and whether there would be uncertainty close to the departure date.

**Itai:**  
Yes. I said that in practice many people are still flying, so I wasn’t treating it as a reason to cancel by default. My main concern was getting stuck in Israel rather than getting there.

We also each mentioned having weddings around that period, which affected the timing.

That part was unrelated to the project.

---

## Action items

### Probe and internal-measurement work

1. **Run a rephrasing robustness check** for the neutral probe and any serious probe variant.
   - Generate semantic-preserving rephrasings of the same questions.
   - Evaluate whether probe performance drops substantially under rephrasing.
   - Use this as a basic stability diagnostic for the probe measurement device.

2. **Construct a framing-agnostic mixed probe training distribution.**
   - Mix incorrect suggestion, suggest-correct, doubt-correct, and corresponding opposite cases.
   - Make the user suggestion uninformative about correctness.
   - Re-train the probe on this mixed distribution and re-evaluate cross-condition generalization.

3. **Rebuild the full train/eval probe matrix** after the above two steps.
   - Report top-1, pairwise ranking metric, and AUC.
   - Check whether any probe now looks plausibly framing-agnostic rather than shortcut-based.

4. **Keep internal claims modest unless a genuinely stable probe is found.**
   - A successful stable probe would support the idea that internal truth evidence remains available across framings.
   - Failure to find one is not decisive evidence of absence.

### External results and writeup

5. **Continue developing the external friction story as a standalone empirical result.**
   - Emphasize movement toward \(b\) rather than flips alone.
   - Show that the effect depends on commitment or confidence.
   - Keep the “same confidence, \(c\) vs \(d\)” comparison central.

6. **Write the external arc clearly.**
   - Probability mass shifts toward the user-backed wrong answer under incorrect suggestion.
   - Movement is smaller when the model is more anchored.
   - At matched confidence, it often does not matter much whether the model initially believed \(c\) or a third answer \(d\).

7. **Show robustness across movement metrics.**
   - Do not rely only on raw probability movement.
   - Include one or more distributional metrics such as KL or related alternatives.
   - Clarify what each metric captures and where interpretation is harder.

### Mechanistic follow-up

8. **Keep pruning as a possible next-stage mechanistic direction.**
   - Read Hadas’s pruning paper once the preprint is available.
   - Think about whether sycophancy may rely on a relatively compact mechanism that could be intervention-tested.
   - Treat this as a follow-up, not as a replacement for stabilizing the current empirical story.

### Logistics

9. **Follow up on GPU access.**
   - Current work is still feasible, but future probe training and pruning experiments will require reliable GPU access.

10. **Keep bookkeeping clean.**
    - Preserve this meeting record.
    - Link it with the earlier April 2 meeting record and the new external analysis artifacts.
    - Update the project summary after the rephrasing and mixed-probe checks.

---

## Short priority list

If the work needs to be prioritized tightly, the three highest-priority next steps are:

1. **Rephrasing robustness check**  
2. **Framing-agnostic mixed probe training**  
3. **Polished external writeup centered on friction, movement, and matched-confidence \(c\)-vs-\(d\) comparisons**

