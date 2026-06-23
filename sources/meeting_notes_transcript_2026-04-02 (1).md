# Meeting Notes and Transcript

**Historical note:** This transcript is retained for discussion history only. Do not use quantitative empirical claims from this transcript as current project results unless they are repeated in `current_empirical_summary_2026-06-20.md`. The main experiment grid was rerun on June 20, 2026.

**Title:** R1 Sycophancy hidden knowledge  
**Date:** April 2, 2026 *(inferred from the filename `02-04-26`; the date format is not fully certain, but this is the most likely interpretation)*  
**Audio file:** `R1 Sychopancy hidden knowledge 02-04-26.m4a`  
**Language spoken:** Hebrew  
**Source transcript:** ivrit.ai noisy transcript  
**Prepared for:** bookkeeping and project record

## Meeting summary

This meeting focused on two related threads in the project:

1. the **friction hypothesis** at the external behavior level, and  
2. the **probe results** as evidence about internal knowledge.

The discussion started by separating **external knowledge** from **internal knowledge**. The external side concerns what can be seen in the model's output probabilities and behavior under user suggestion. The internal side concerns whether the model still internally represents the truth after biased framing.

### Main discussion points

#### 1. Friction hypothesis
The main candidate insight is that sycophantic movement appears to be stronger when the model is **less confident** in its original answer. The important point is that this is not only about full flips on near-boundary examples. Instead, the observed effect is that the *magnitude of movement toward the user-suggested answer* is larger when the model starts from weaker commitment.

The interpretation discussed was deliberately cautious:
- the result may suggest that models are easier to influence when they are uncertain or do not know
- this may connect sycophancy to competence or knowledge, but it is too early to claim that sycophancy is simply a capabilities problem
- a stronger model may be harder to move, but that remains a hypothesis rather than a conclusion
- the relationship is unlikely to be a simple linear one

A key distinction was emphasized: this evidence is currently **external only**, meaning it comes from the output distribution rather than from internal-state analysis.

#### 2. A three-level framework for interpreting probe results
A central part of the meeting was an attempt to organize what kinds of claims probe results can actually support.

The proposed framework had three levels:

- **Level 1:** A probe trained in one condition, such as neutral framing, still performs nontrivially when tested in another condition, such as biased framing. This supports the claim that there is still usable signal.
- **Level 2:** Performance stays roughly similar across conditions. This would suggest some robustness to framing, though not identity of the underlying representation.
- **Level 3:** The probe behaves like the same function across framings, effectively acting as a framing-agnostic truth detector. This is the strongest and idealized version.

The conclusion in the meeting was that the results do **not** support level 3, and only partially approach level 2 in some settings. However, they do support level 1.

#### 3. Distribution shift versus real internal change
A major interpretive concern was whether performance drops across framings reflect:
- a real change in the model's internal representation, or
- probe failure under distribution shift

To address this, the meeting returned to the **congruent suggestion** control. In that condition, the user suggestion matches the answer the model would already have given. This serves as an out-of-distribution test that should not create the same kind of substantive internal conflict as an incorrect suggestion.

The main interpretation discussed was:
- when the suggestion is congruent with the model's prior answer, the probe behaves much more stably
- when the suggestion is incongruent, the probe behavior degrades more substantially
- this pattern supports the idea that at least part of the effect is due to a real internal representational change rather than mere probe brittleness

#### 4. Implications for how to think about sycophancy
One tentative conclusion raised in the meeting was that sycophancy may look less like a clean tradeoff between:
- wanting to be correct, and
- wanting to please the user

and more like a phenomenon where the main axis is simply whether the model **knows** or **doesn't know**. On this view, user-pleasing may be secondary, while uncertainty or weak knowledge is primary.

This was not treated as settled. Hadas explicitly raised the possibility that even apparently strong probe results may still admit alternative interpretations, including the possibility that the probe is using conditional heuristics rather than recovering a clean truth representation.

---

## Things we said to do for next meeting

1. **Position the friction hypothesis relative to the literature.**  
   Check more carefully whether this result, or something close to it, has already appeared.

2. **Avoid overclaiming.**  
   Be careful not to conclude too quickly that sycophancy is purely a capabilities problem.

3. **Compare model competence levels.**  
   Test whether weaker versus stronger models differ systematically in susceptibility to user suggestions.

4. **Characterize the functional form of the friction effect.**  
   Investigate whether the confidence-to-susceptibility relationship is linear, thresholded, saturating, or otherwise structured.

5. **Return to the friction result after probe analysis.**  
   Treat it as a real signal, but one that still needs tighter framing and stronger evidence.

6. **Use the three-level framework to organize probe claims.**  
   Present probe findings in terms of level 1, level 2, and level 3 rather than only raw tables.

7. **Complete and inspect the train/test condition table.**  
   Systematically compare neutral-trained and biased-trained probes across evaluation conditions.

8. **Keep separating distribution shift from internal change.**  
   Do not interpret cross-condition degradation too quickly as evidence about hidden knowledge.

9. **Use the congruent-suggestion control as a sanity check.**  
   Keep using it to test whether drops are due to OOD shift alone or to something more substantive.

10. **Think about additional probe constructions.**  
    Hadas mentioned another possible probe setup or alternative construction that may be worth revisiting.

11. **Inspect activations or probe inputs more directly.**  
    Compare how similar the probe inputs or activations are across neutral, congruent, and incorrect-suggestion conditions.

12. **Be cautious about interpreting robust cross-condition performance.**  
    Even if a probe trained on incorrect suggestion generalizes to neutral, that does not by itself prove it learned a clean truth representation.

13. **Look for alternative interpretations.**  
    In particular, ask whether the model may still know the truth after bias while the probe is exploiting prompt-dependent heuristics.

---

## Cleaned English transcript

**Note:** This is a reconstructed English transcript from a noisy Hebrew ASR output. Speaker identities are inferred from context and may not be exact.

- **Speaker 1 = likely Itai**
- **Speaker 0 = likely Hadas**

### Transcript

**Speaker 1:**  
I think we should split this into two things. There’s hidden knowledge, and there’s external knowledge. On the external side, I think we already have a decent handle on what’s going on. We can demonstrate sycophancy. And now the question is really about the internal knowledge and what exactly to do with it. I think we now have a better picture of where things stand, and I want to plan another round of experiments, or maybe think about it differently. But this isn’t like previous times where I say, “Here, I did another data sweep, let’s talk about it.”

**Speaker 0:**  
Yes.

**Speaker 1:**  
Okay. So let’s start with the first insight, which I don’t think has really been done in the literature before. Sycophancy itself, fine, people have shown that. But what we’re calling the friction hypothesis, I think that’s new, as far as I know. And I think it’s nontrivial, and that there really is something there, so we should talk about how interesting it is. What we’re noticing is that it’s easier to move the model away from the answer it originally believed and toward the answer suggested by the user when the model is less confident in its own answer.

**Speaker 0:**  
Yes, right.

**Speaker 1:**  
In other words, it’s not just that it’s easier to get a full flip from correct to incorrect because the item is near the margin, everything is close, everything is balanced, and then I push it a little and it fully flips. Rather, if I actually look at the push itself, the shift from \(P(\text{correct})\) to \(P(\text{bias})\), I see a real movement there. So it’s not that every question gets the same shift and then low-confidence items just happen to flip more easily. It’s that I really have to push harder when the model is more confident. So I think this is some preliminary insight. It’s not yet clear what else to do with it, but it seems like a real signal.

**Speaker 0:**  
Yes, that’s interesting to me too.

**Speaker 1:**  
I did a quick literature search using ChatGPT, just to see whether there’s anything close to this. It looks like there are things in the neighborhood, but not this exactly. The tempting thing to say is that a lot of sycophancy comes from lack of understanding, and that if the model were more capable it would be less sycophantic, but that may be too strong. A safer version would be: it’s easier to influence the model when it is uncertain, or when it doesn’t know. And we could do more experiments specifically on that. For example, what happens if I compare a very weak model to a highly capable one? Do I see some systematic difference?

**Speaker 0:**  
Right, but that’s still only external, right? Just the probabilities.

**Speaker 1:**  
Yes, only the external side.

**Speaker 0:**  
Only the probabilities.

**Speaker 1:**  
Yes. It also helps separate, conceptually, what we know externally from what we know internally.

**Speaker 0:**  
Okay.

**Speaker 1:**  
And on the internal side, we really need to talk a lot more now, because as we saw last time, it’s very delicate.

**Speaker 0:**  
Yes.

**Speaker 1:**  
But externally, this is an interesting fact, and we should think about how to develop it. The nicest thing to be able to say would be that sycophancy is a capabilities problem, and once you solve the capabilities problem, sycophancy goes away. That would be an amazing claim. We don’t know if that’s true yet, but in some sense that is what this hypothesis is pointing toward: if the model is more confident in its answer, then it is harder to move. And by “confident,” I don’t necessarily mean confident in the correct answer, just confident in some answer.

**Speaker 0:**  
Right. The question is whether we’re actually convinced that in cases where it’s maximally confident, it really becomes very hard to move.

**Speaker 1:**  
How do we know that such a regime even exists?

**Speaker 0:**  
Because the question is whether the relationship is something like linear.

**Speaker 1:**  
It’s not linear. It’s not clear what the functional form is. We’ve seen how hard it is to characterize these things cleanly.

**Speaker 0:**  
Right.

**Speaker 1:**  
So we should put an asterisk on that.

**Speaker 0:**  
So the statement “if it’s really confident, then we won’t be able to move it” is not necessarily true.

**Speaker 1:**  
Right. We’ll be able to move it less. It’ll just be harder. That’s the friction interpretation, it’s harder to push it. So I think the right interpretation is that this is some signal we’re seeing in the data from the experiments we already have, and it’s not fully worked out yet. There’s more to do here. But it’s something external, not internal, that I don’t think people really know about, and it’s worth thinking about how to develop it.

Okay, let’s keep going and maybe come back to it later.

Now let’s talk about the probe. It’s really hard to analyze this properly, and even to think clearly about what to do. Conceptually, I’m thinking about it as three levels of claims we could make about the probes.

**Speaker 0:**  
Meaning?

**Speaker 1:**  
What got mixed up for us last time was that with the probe, there are two different issues: what dataset it was trained on, neutral, biased, before bias, after bias, and then what its performance is before or after the bias. What we would ideally want is a probe that is agnostic to framing: I train it on neutral, it works, and the truth signal it has internally survives even after I inject bias into the prompt. In other words, after I add the bias, nothing changes. The model still “knows” whether the answer is correct or incorrect. That is the property we would want. Let’s call that level 3.

So level 3 means: I trained a probe that knows how to identify true versus false, and the score it gives to correct versus incorrect is independent of how the user framed the question. That’s the best-case scenario.

The easiest thing we can say is just that we can train probes and test them on held-out data, and they work. There is hidden signal there, regardless of the bigger hidden-knowledge question, the probe can distinguish pretty well between what is true and what is false. I think we can say that. That is: I can train a probe on neutral data, evaluate it on neutral or on biased prompts, and it gets nontrivial performance, well above chance. More than just “a bit above chance,” the performance is decent.

**Speaker 0:**  
Okay.

**Speaker 1:**  
So in terms of raw performance, we know that there is some signal. Then there’s a middle level, which is to say that after I switch to the biased condition, I still get roughly the same performance. That is different from saying that literally nothing changes after I add the framing. The strongest version would be that everything is exactly unchanged. The middle version is just: performance stays about the same.

**Speaker 0:**  
Okay, I’m following.

**Speaker 1:**  
Right. So I’m defining three levels of what we would expect from a probe.

**Speaker 0:**  
Okay.

**Speaker 1:**  
Level 1: I train a probe on neutral, and it more or less still works on the biased prompts. What does “works” mean? It means it can still tell what is right and what is wrong at a level that is better than trivial.

Level 2: when I look at the biased prompts, I still get overall pretty good performance, basically the same performance. There isn’t much of a drop after I add the bias.

Level 3: it gives literally the same score per question.

**Speaker 0:**  
Okay. The first one is with no bias in either case? And the second is after you introduce the bias—

**Speaker 1:**  
No. I mean that even if I test on the biased prompts, it still performs okay. We can just look at the numbers. Suppose I train only on neutral, and then I test on the biased prompts. What I’d want to see is that overall performance, say top-1 accuracy, is still better than trivial. For example, I have 93% here, and with incorrect suggestion it drops to 65%. That means that 65% of the time it still identifies the correct answer.

**Speaker 0:**  
Yes, okay.

**Speaker 1:**  
And that’s the pairwise or ranking-style metric we talked about. So level 1 is just saying: the performance here is still decent.

**Speaker 0:**  
It’s doing something.

**Speaker 1:**  
Exactly. It’s doing something nontrivial. There is a strong signal there. Then claim number 2, level 2, is to say: yes, but it’s basically the same here and here.

**Speaker 0:**  
You want to say it’s the same?

**Speaker 1:**  
If the score were the same, that would be level 2. And level 3 is to say: it’s literally the same score on each question. Level 3 is: my probe is a truth machine.

**Speaker 0:**  
My probe doesn’t change at all, yes.

**Speaker 1:**  
Exactly. It’s a truth machine. Regardless of how the user frames the question, it tells me whether the model internally knows or doesn’t know the truth.

**Speaker 0:**  
Right.

**Speaker 1:**  
So we are not at level 3. That’s not what I’m claiming. And for level 2, there is some drop, but it’s a small drop.

**Speaker 0:**  
But even that small drop, we don’t know whether it’s a drop in the probe, or a drop in the model’s actual knowledge. The probe is only a proxy for the model’s knowledge. So how does that fit in?

**Speaker 1:**  
You mean: maybe there is a real internal change, and the external shift reflects that?

**Speaker 0:**  
Not exactly. I mean that the probe is only a proxy for the internal state, and all kinds of things could affect it. So once we see a drop, the question, and this was really my question last time, is whether the drop, say from 81 to 65, is because the model really knows less, meaning it now actually thinks the incorrect suggestion is correct, or whether it’s just that the probe generalizes badly out of distribution.

**Speaker 1:**  
Right. So the thing you raised last time, let’s touch that, because it’s interesting. I added another condition. I called it “congruent,” or overlapping bias.

**Speaker 0:**  
Does “congruent” really mean overlap?

**Speaker 1:**  
I’m translating from Hebrew on the fly. I’m thinking in Hebrew. But yes, overlapping or congruent. The idea is: before, we were giving an incorrect suggestion. Now I’m not suggesting the correct answer either. Instead, I suggest the answer that the model itself already believes. So first I sample from the model what answer it wants to give, what it thinks, even if the confidence isn’t very high. Then I inject that answer as the user suggestion.

**Speaker 0:**  
So you’re injecting something that shouldn’t really change the model’s beliefs.

**Speaker 1:**  
Right.

**Speaker 0:**  
If anything it should only reinforce them, make it more confident.

**Speaker 1:**  
Exactly. But now I can test whether the probe behaves differently when I change the evaluation distribution, because this is still out of distribution. I trained on one kind of prompt, and now I’m evaluating on another kind.

**Speaker 0:**  
Ah, so you’re only changing the test data.

**Speaker 1:**  
Yes, exactly. That was your hypothesis last time, so I checked it.

**Speaker 0:**  
I think I had two ideas, I don’t remember, but yes.

**Speaker 1:**  
I wanted to complete the table and see what happens when I change each piece.

**Speaker 0:**  
Right, we’ll come back to the table. But I also had another idea about how to do it, instead of A and B, maybe do C. But never mind for now.

**Speaker 1:**  
Yes, yes.

**Speaker 0:**  
That was just another idea for how to build a probe.

**Speaker 1:**  
Right. But again, at a high level, because it’s very easy to get confused here between different meanings, what I want to say is: is the probe good or not? And I don’t want the answer to be a giant pile of numbers and tables. I want it to be these three levels. Is there a drop when I move out of distribution? And if there isn’t really a drop in performance, can I say it’s basically the same thing per question? In other words: is it agnostic to framing?

**Speaker 0:**  
Okay, okay.

**Speaker 1:**  
That’s really the question.

**Speaker 0:**  
But the moment there is any drop, I immediately start treating that drop as the problem. That’s what bothers me.

**Speaker 1:**  
Yes. I’m trying to think of a good analogy. Level 1 says: the model is still okay even after I add the bias, better than trivial. Level 2 says: the performance is still overall similar. Level 3 says: it is literally the same function. Those are the three levels.

**Speaker 0:**  
But with “the same,” what exactly do you mean?

**Speaker 1:**  
I mean it represents exactly the same function.

**Speaker 0:**  
Okay.

**Speaker 1:**  
Level 2 means the drop is only small.

**Speaker 0:**  
Okay.

**Speaker 1:**  
Okay?

**Speaker 0:**  
Fine, I’m with you.

**Speaker 1:**  
Also, it doesn’t matter whether these levels are mutually exclusive. That’s not the point. This isn’t even really for the paper. It’s more for us, as a way of analyzing things.

**Speaker 0:**  
Okay.

**Speaker 1:**  
Because all of this is really confusing. Honestly, I should probably draw it as a spectrum. The ideal case is that it’s the same function. That’s what we would want ideally. I don’t think we can say that, but—

**Speaker 0:**  
Wait, show me the actual results.

**Speaker 1:**  
Okay. So now we’re trying to answer exactly that question.

**Speaker 0:**  
You have a probe trained on neutral. It’s still trained on neutral, right? And now you’re evaluating it on a prompt where the suggestion is the same answer the model would have given anyway.

**Speaker 1:**  
Right.

**Speaker 0:**  
What do you see?

**Speaker 1:**  
Then the behavior is basically the same. Or maybe let’s start from the probe perspective: once I add this congruent bias, I basically get the same function.

**Speaker 0:**  
The accuracy here, is that accuracy relative to the earlier predictions?

**Speaker 1:**  
What do you mean?

**Speaker 0:**  
Relative to the probe’s earlier prediction.

**Speaker 1:**  
Yes, in the sense of how often it changes its answer. And the answer is: almost never. It mostly just reinforces what it already believed. But I think the stronger statement is not just that it keeps the same answer, it’s that what it believed before is basically preserved.

**Speaker 0:**  
Yes.

**Speaker 1:**  
That is, the “correct” score is about the same, the separation is about the same. And then compare that to the incorrect-suggestion condition.

**Speaker 0:**  
Ah.

**Speaker 1:**  
The numbers there are much higher.

**Speaker 0:**  
What do you mean “higher”? Higher compared to what?

**Speaker 1:**  
Compared to the condition where I suggest the answer the model already believed.

**Speaker 0:**  
Which numbers are higher?

**Speaker 1:**  
The percentage of cases where it changes. The change rate.

**Speaker 0:**  
Yes.

**Speaker 1:**  
So what do I get from that? That the probe is functioning pretty well even under this condition. It’s basically the same function.

**Speaker 0:**  
Yes, I agree. That means the model really changes its belief.

**Speaker 1:**  
Yes. And it’s also a good sanity check that the probe was trained properly.

**Speaker 0:**  
Yes.

**Speaker 1:**  
But when I inject an incongruent bias, an incorrect suggestion, that confuses it, and it changes the internal representation.

**Speaker 0:**  
Its internal representation really changes.

**Speaker 1:**  
Yes. I think that’s the conclusion.

**Speaker 0:**  
I think that actually strengthens the interpretation that sycophancy is incompetence. The model really believes the user’s suggestion.

**Speaker 1:**  
Yes. I think that’s what’s happening. It puts serious pressure on the picture I had in my head, which was that sycophancy is a combination of two forces: the force that wants to be correct, the accuracy objective it was trained on, and the force that wants to be user-pleasing, because that’s part of the RLHF pressure. Here it looks like maybe that’s not the right picture. It seems more like it’s fundamentally about whether the model knows or doesn’t know. That’s the main axis. The user-pleasing component looks secondary here, at least in this setting.

**Speaker 0:**  
Yes. But with the congruent suggestion, how can it be that it gets stronger there, from 81 to 88, for example?

**Speaker 1:**  
It’s not really a dramatic strengthening. It’s roughly the same.

**Speaker 0:**  
81 and 88.

**Speaker 1:**  
It should strengthen a little, because I’m reinforcing it. I’m telling it, “Yes, that’s the right answer.” Again, this is not like the “suggest correct” condition, where I’m suggesting the actually correct answer even when the model doesn’t know the correct answer. Here I’m just reinforcing whatever it already believed anyway. It’s kind of an echo chamber, maybe that’s actually the right term. So it’s interesting that it’s roughly the same. The first row and the last row are about the same.

And that was the row you asked for last time. That was the one you really wanted. So now I’m also changing which probe I’m looking at.

**Speaker 0:**  
Because now the probe is trained on incorrect suggestion.

**Speaker 1:**  
Yes. Before, we only looked at the neutral probe and asked how it changes. Now we’re also looking at a probe trained on incorrect suggestion.

So: neutral probe on neutral test, that’s in-distribution test, AUC 93%, basically perfect.

**Speaker 0:**  
Wait, which probe again?

**Speaker 1:**  
The probe trained on neutral and tested on neutral.

**Speaker 0:**  
Okay.

**Speaker 1:**  
On held-out neutral test data. So it’s in-distribution, but still test data.

**Speaker 0:**  
Yes.

**Speaker 1:**  
And it’s almost perfect.

**Speaker 0:**  
Okay.

**Speaker 1:**  
Then I change the evaluation distribution and test it on biased prompts, and it drops.

**Speaker 0:**  
Right.

**Speaker 1:**  
Now I do the same thing on the other side: incorrect-suggestion probe on incorrect-suggestion test. Again, very, very high. In other words, if I train a probe on the biased condition and test it on the biased condition, I still get very good performance.

**Speaker 0:**  
Yes, but in some cases it’s lower than neutral-on-neutral, which is surprising.

**Speaker 1:**  
Is it consistently lower? It looks pretty similar overall.

**Speaker 0:**  
No, here it’s weaker. Or maybe it depends on the dataset.

**Speaker 1:**  
They look very similar to me.

**Speaker 0:**  
What surprises me is that there are even cases where it’s weaker. I would have expected it always to be better. I mean, literally, I’m giving it the answer in the prompt.

**Speaker 1:**  
Right. In principle it only has to learn something like: when the user says, “I think the answer is X,” just read X.

**Speaker 0:**  
Exactly. It could almost learn a trivial rule.

**Speaker 1:**  
Yes. But at the same time, the model also has some internal truth signal that it believes, and that should also show up somewhere.

**Speaker 0:**  
No, this is really just a question of what the probe learned.

**Speaker 1:**  
Yes. It would actually be interesting to look at the activations here. I haven’t done that yet. For example: are the activations here and there similar? I’m sure they’re highly correlated.

**Speaker 0:**  
Activations in what sense?

**Speaker 1:**  
I mean the input to the probe, how different the probe input is.

**Speaker 0:**  
Ah, you mean what information is even available for the probe to learn.

**Speaker 1:**  
Exactly.

**Speaker 0:**  
Yes. But even if the activations are similar, it still has access to the incorrect-suggestion information.

**Speaker 1:**  
Right. So in principle it should be easier.

**Speaker 0:**  
Exactly. I’m not even saying it necessarily would learn that shortcut, but it could have. In principle it could reach 1.0 almost trivially.

**Speaker 1:**  
Yes. If you gave me a really hard exam and I knew none of the material, I could still get 100% if you also gave me a perfectly reliable rule.

**Speaker 0:**  
Yes.

**Speaker 1:**  
Exactly.

**Speaker 0:**  
Okay, but we also see that the difference between neutral and suggestion isn’t really consistent. It’s more or less the same story.

**Speaker 1:**  
Between neutral and suggestion, in what sense?

**Speaker 0:**  
In the suggestion condition. I’m looking at the middle values there, like 89 and 90.

**Speaker 1:**  
Right.

**Speaker 0:**  
There’s only a very small difference. Maybe that means it captures a bit of bias, but—

**Speaker 1:**  
Yes, more than here.

**Speaker 0:**  
What do you mean “more than here”?

**Speaker 1:**  
I mean that moving out of distribution from neutral to bias seems to have a bigger effect than moving out of distribution in the other direction.

**Speaker 0:**  
Yes.

**Speaker 1:**  
Let me tell you my overall conclusions from this whole probe exercise, and then we can talk about them. I also plotted what we already saw before, how things change after I inject this congruent or overlapping bias. If the model was correct before, then after I add congruent bias it gets slightly better. If it was wrong before, then after I add congruent bias it gets slightly more wrong. And if I compare that to the incorrect-suggestion condition, it behaves differently.

**Speaker 0:**  
By “it,” do you mean the probe?

**Speaker 1:**  
Yes, the probe.

**Speaker 0:**  
What exactly do you mean?

**Speaker 1:**  
I mean: if the model was correct, and I suggest the answer it already believed, then it gets a bit stronger. But if instead I suggest an incorrect answer, performance drops, and it becomes much more likely to flip. Which is exactly what you’d expect.

**Speaker 0:**  
If suggesting the incorrect answer had made it improve, that would have been very strange.

**Speaker 1:**  
No, no, in that plot, “higher” was better in a different sense.

**Speaker 0:**  
Right, you mean the one next to it.

**Speaker 1:**  
Yes, they were all shown together.

Anyway, the conclusions I’m drawing from all this—

**Speaker 0:**  
But there is one interesting point here. Sorry, go ahead.

**Speaker 1:**  
No, go ahead.

**Speaker 0:**  
There is one interesting point: this probe, which overall I don’t really like, still seems interesting. Go back to the table. It doesn’t have that large drop.

**Speaker 1:**  
Yes, exactly.

**Speaker 0:**  
Yes.

**Speaker 1:**  
Yes.

**Speaker 0:**  
Because the probe trained on incorrect suggestion also works on neutral. So the question is—

**Speaker 1:**  
Meaning?

**Speaker 0:**  
I’m trying to think whether there’s some alternative interpretation under which the model still knows the truth, even after the bias.

**Speaker 1:**  
Yes. And just to open a parenthesis: level 2 would be to say, “Look, the performance is roughly the same, around 90%.” But that still does not mean it is literally the same linear function. It doesn’t mean it is the same model.

**Speaker 0:**  
Yes.

**Speaker 1:**  
Again, what we would want ideally is that the whole table be the same. There should be no difference between this row and that row. That would be the ideal scenario.

**Speaker 0:**  
Let me make sure I understand. This probe that you trained on incorrect suggestion, it succeeds on neutral too.

**Speaker 1:**  
Yes.

**Speaker 0:**  
It looks quite robust.

**Speaker 1:**  
It is still a bit worse than the neutral-trained probe, yes.

**Speaker 0:**  
Fine, but I trust its results more.

**Speaker 1:**  
Because the difference between this and that is smaller?

**Speaker 0:**  
Because it had the opportunity to learn the trivial heuristic from the prompt, but apparently it learned something else.

**Speaker 1:**  
Yes. The fact that it still works on neutral suggests it isn’t just doing the obvious prompt heuristic.

**Speaker 0:**  
If it were just doing that, it would get 1.0.

**Speaker 1:**  
Exactly. That’s not what’s happening. It really did learn something about the answer.

**Speaker 0:**  
And then I want to say: “Look, the model still knows that the user-backed answer is wrong, even when it verbally goes along with it.” But then the question is: okay, I still don’t have proof that that is really what the probe learned, rather than some other heuristic. How do I know it didn’t learn two separate rules? Maybe it learned: when the prompt looks like this, use one heuristic. When the prompt doesn’t, use another. I still don’t have a way to prove otherwise.

---

## Brief extracted action list

For convenience, here is the shortest version of the follow-up list:

- check literature positioning for the friction hypothesis
- avoid overclaiming that sycophancy is just a capabilities problem
- compare weaker and stronger models
- characterize the functional form of the friction effect
- organize probe claims via the three-level framework
- complete the cross-condition probe table
- separate OOD probe failure from genuine internal change
- keep using the congruent-suggestion control
- revisit alternative probe constructions
- inspect activations or probe inputs more directly
- be cautious about interpreting robust cross-condition probe results
- keep testing alternative interpretations where the model may still know the truth after bias
