# Hidden Factual Knowledge Methodology

## Purpose

This note extracts the technical methodology from **Inside-Out: Hidden Factual Knowledge in LLMs** and rewrites it in a form that is easier to reuse when designing **hidden knowledge in sycophancy** experiments.

It focuses on four things:

1. What the original paper actually measured.
2. The exact experimental pipeline.
3. The representation and probe details that matter for replication.
4. How to port the method into the sycophancy proposal.

---

## 1. What the original paper is testing

The paper is **not** asking only whether the model answers a question correctly.

It asks a stronger question:

> For a given question, can the model rank correct answers above plausible wrong answers better from its internal activations than from its observable token probabilities?

So the core object is **answer ranking**, not single-output accuracy.

### Core definitions

For one question `q`, define:

- `A(o)`: the set of correct answers for the gold object `o`
- `Ã(o)`: the set of plausible candidate answers, both correct and incorrect
- `S(q, a)`: a scoring function for answer `a`

Then the per-question knowledge score is:

`K_q = fraction of (correct answer, wrong answer) pairs where S(q, correct) > S(q, wrong)`

The dataset-level score is the average of `K_q` across questions.

There is also a strict version:

- `K* = 1` only if **all** correct answers beat **all** wrong answers for that question
- otherwise `K* = 0`

### What counts as hidden knowledge

They compare:

1. **External scoring**
   Uses only token-level probabilities or verification probabilities that are observable from the model output process.

2. **Internal scoring**
   Uses intermediate activations, specifically a linear probe on hidden states.

The model shows hidden knowledge if the internal scorer ranks answers better than every tested external scorer.

---

## 2. End-to-end pipeline in the hidden factual paper

## Step 1. Build a factual QA dataset

### Source dataset

They start from **EntityQuestions**, which converts Wikidata triples into QA pairs.

### Relations used

They keep only relations that are both:

1. hard to guess
2. close to single-answer and easy to grade

The four relations are:

- `P26`: spouse
- `P176`: manufacturer
- `P264`: record label
- `P50`: author

### Split construction

- Test and dev come from the EntityQuestions **test** split
- Train comes from the EntityQuestions **train** split
- They target **500 questions per relation**
- They reserve **10% of the remaining test items for dev**
- Train and dev are used only for the internal scorer

### Important filtering

They remove:

- questions with more than one gold answer
- questions whose text already contains the gold answer
- duplicate questions

To stop leakage between probe train and test, they also remove overlapping train and test facts. For relations where subject and object share the same type, such as spouse, they also block subject and object overlap across train and test.

---

## Step 2. Build the candidate answer set for each question

This is the most important part of the paper.

They do **not** compare the gold answer only against random negatives. They build a large candidate set of plausible answers from the model itself.

### Candidate generation

For each question:

1. Generate **one greedy answer**.
2. Generate **1,000 sampled answers** with temperature `1`.
3. Deduplicate into a set of unique answer strings.
4. If the gold answer was never sampled, manually add it.

The paper reports that the gold answer is missing from the sampled set in about **64%** of questions on average.

### Why they do this

Uniform random negatives would be too easy. Model-sampled answers are much harder negatives because they come from the model's own belief distribution.

### Question paraphrases

They do **not** paraphrase the question in the main experiment.

They use the single original question only, so `|Q(s,r)| = 1`.

---

## Step 3. Label which candidate answers are correct

They estimate the set of correct answers with an **LLM judge**.

### Judge model

- `Qwen2.5-14B-Instruct`

### Judge behavior

For non-exact-match answers, the judge compares:

- question
- gold answer
- proposed answer

It outputs one of four labels:

- `A`: correct
- `B`: incorrect
- `C`: wrong gold
- `D`: error

The prompt is relation-specific and program-guided. It acts like a small decision tree.

### Important implementation choice

If a candidate answer triggers a judge error, they filter out the **entire question**, not just that answer.

### Another filter

If all sampled answers are correct, they discard that question because there is no ranking problem left.

---

## Step 4. Compute external scores

They test three external scoring methods.

### 4.1 Production score: `P(a|q)`

This is the autoregressive probability of the whole answer string.

For an answer with tokens `a_1 ... a_n`:

`P(a|q) = Π_i P(a_i | q, a_<i)`

### 4.2 Length-normalized production score: `P_norm(a|q)`

This is the geometric mean token probability of the answer.

Equivalent form:

`P_norm(a|q) = exp(mean token logprob)`

### 4.3 Verification score: `P(True)` in the paper, implemented as `P(A)` over `{A,B}`

Conceptually this score asks the model whether the candidate answer is correct.

In the appendix implementation, the model sees a verification prompt like:

- Question: `{question}`
- Proposed Answer: `{answer}`
- `A: CORRECT`
- `B: INCORRECT`

Then they compute the next-token probability of `A`, normalized only over the two labels `A` and `B`.

### Important detail for replication

For `P(a|q)` and `P_norm(a|q)`, they do **not** just concatenate question and answer text. They construct the full chat sequence exactly as if the model had generated the answer after the prompt, including the expected special tokens. Then they run a forward pass and read the answer-token logits.

This matters a lot for scoring the manually added gold answer, which may never have been sampled.

---

## Step 5. Compute the internal score from hidden activations

This is the internal scorer `T_M`.

### Model input to the probe

For each `(q, a)` pair, they create a sequence that simulates the model generating `a` after `q`.

They then extract a hidden representation `h_M(q, a)` from one layer and train a **linear logistic regression classifier** to predict whether `a` is correct.

The probe output probability is the internal score.

### What is explicit in the paper

- one probe per layer
- logistic regression objective
- input is the hidden state for the `(q, a)` sequence
- layer chosen by dev performance
- upper layers work better than early layers
- performance usually improves around layers `11 to 12` out of `32` and then stabilizes

### What is **not fully pinned down** in the paper text

The paper text available here says `h_M(q, a)` is a single hidden state from the `(q, a)` sequence, but it does **not clearly specify** which answer-token state is used when the answer has multiple tokens.

So for a faithful reimplementation you must explicitly choose and report:

1. the token positions that count as the answer span
2. the pooling rule across answer tokens, if any
3. the exact hidden stream you read, such as post-block hidden state or residual stream

A practical choice, and the one that best matches the paper's notation of a single hidden state, is:

- **use the final answer token hidden state for each layer**

The sycophancy proposal already makes this ambiguity explicit by defining a pooled representation `φ_ℓ(x, ŷ)`, for example the final token or the mean of the last few answer tokens.

---

## Step 6. Train the probe in a knowledge-aware way

This is another very important methodological choice.

They do **not** train the probe on arbitrary correct and incorrect answers.

Instead they try to train on questions where the model likely **knows** the answer, so the hidden state should contain useful truth information.

### Probe training data construction

From the train split:

1. Keep only questions where the model's **greedy answer exactly matches the gold answer**.
2. Treat that greedy correct answer as the **positive** example.
3. Sample **200 additional responses** at high temperature.
4. Keep sampling until an incorrect answer appears.
5. Use one incorrect answer from the same question as the **negative** example.

This gives paired data of the form:

- same question
- one correct answer
- one plausible wrong answer

The paper text around the appendix merges the footnote number into the temperature value. The intended training-time temperature is `2`, with a footnote explaining that higher temperature helps surface wrong answers when greedy is already correct.

### Final probe train size

- `2,000` training examples total
- `500` per relation
- they use a single cross-relation probe, not one probe per relation

### Why this matters

This makes the probe learn **which answer is correct for a known question**, not just whether the model looks globally uncertain.

---

## Step 7. Evaluate hidden knowledge

For each question and each scorer:

1. score every candidate answer
2. compare all `(correct, wrong)` answer pairs
3. compute `K_q`
4. average over questions to get `K`
5. compute `K*` for perfect ranking

Then compare:

- best external scorer
- internal probe scorer

They use paired statistical testing after binning the test set into subsets.

---

## Step 8. Run two additional checks

### 8.1 Deeply hidden knowledge check

They add the gold answer even when it was never sampled, then ask:

> Can the probe rank the gold answer above all wrong candidates even though the model never generated it in 1,000 tries?

This is the strongest version of the claim that the model internally knows more than it externally says.

### 8.2 Test-time reranking check

They sample many answers and ask whether ranking them with the probe beats:

- greedy decoding
- random choice
- majority answer
- ranking by `P(a|q)`

This tests whether the internal score is useful for downstream answer selection.

---

## 3. What you need to replicate the hidden factual setup

## Required ingredients

1. **Models**
   - Llama-3-8B-Instruct
   - Mistral-7B-Instruct
   - Gemma-2-9B-It
   - optionally Qwen3-32B for a larger-model check

2. **Dataset**
   - EntityQuestions
   - relations `P26`, `P176`, `P264`, `P50`

3. **Generation code**
   - greedy decode once per question
   - sample 1,000 times at temperature `1`
   - preserve exact chat template and special tokens

4. **Judge code**
   - exact-match shortcut first
   - Qwen2.5-14B-Instruct fallback judge
   - relation-specific grading prompts

5. **Scoring code**
   - answer logprob `P(a|q)`
   - length-normalized logprob `P_norm(a|q)`
   - verification probability `P(A | q, a)` over labels `{A,B}`

6. **Probe code**
   - hidden-state extraction for `(q, a)` sequences
   - one logistic regression probe per layer
   - dev-based layer selection

7. **Evaluation code**
   - candidate deduplication
   - correct and wrong pair construction
   - `K` and `K*`
   - significance testing

## Open implementation decision

The largest under-specified item is the exact answer-token representation fed to the probe. The public GitHub repository linked from the paper currently has only a README saying that code will be uploaded later, so this detail is not resolved there yet.

---

## 4. How this maps into the sycophancy proposal

The proposal keeps the same basic logic but changes the perturbation.

In the hidden factual paper, the perturbation is:

- compare external scores and internal scores over candidate answers for a neutral factual question

In the sycophancy proposal, the perturbation becomes:

- compare a **neutral prompt** `x = q`
- against a **bias-injected prompt** `x' = inject(q)` that pushes the model toward a false user stance

## Direct transfer from hidden factual to sycophancy

### Data objects

For each QA item:

- neutral prompt `x_i = q_i`
- bias-injected prompt `x'_i = inject(q_i)`
- neutral model response `ŷ_i`
- injected model response `ŷ'_i`
- gold short answer `a_i`
- correctness labels `z_i, z'_i`

### Internal signal

At layer `ℓ`, define a pooled activation `φ_ℓ(x, ŷ)` over answer tokens.

Then fit a linear probe:

`p_ℓ(x, ŷ) = sigmoid(w_ℓ^T φ_ℓ(x, ŷ) + b_ℓ)`

The proposal already states this explicitly.

### Main questions to test

1. **Stability under pressure**
   Does internal correctness evidence stay high after bias injection, even when the output becomes wrong?

2. **Confident-yet-compliant failures**
   Among sycophantic wrong answers, how often is the internal score still high?

3. **Interaction regime effects**
   Do single-turn and multi-turn challenge prompts change internal evidence in the same way?

## Best methodological choice if you want to stay very close to the hidden factual paper

Use a **candidate-answer ranking version** of the sycophancy experiment, not only a response-level correctness classifier.

For each bias-injected prompt `x'`, compare at least:

- the gold answer
- the model's sycophantic answer
- other plausible sampled answers

Then compute the same pairwise ranking metrics `K` and `K*`.

This keeps the experiment aligned with the original hidden factual methodology and avoids collapsing everything into a single correctness score.

---

## 5. Recommended design decisions for new code

If the goal is to build a robust codebase for similar experiments, fix these choices up front and log them in every run:

1. **Prompt template version**
   Chat template and all special tokens must be frozen.

2. **Answer normalization**
   Decide how to strip whitespace, punctuation, and chat suffixes.

3. **Candidate deduplication rule**
   Deduplicate before scoring.

4. **Hidden-state extraction rule**
   Prefer final answer token first. Add pooled variants later as an ablation.

5. **Layer search rule**
   Train one linear probe per layer. Pick layer on dev only.

6. **Judge fallback policy**
   Decide whether a judge error drops the answer or the whole question. The paper drops the whole question.

7. **Leakage checks**
   Block subject and object overlap between probe train and probe test.

8. **Bias injection families**
   Keep single-turn and multi-turn challenge prompts separate.

9. **Metric family**
   Track both response-level accuracy and pairwise ranking metrics.

10. **Stress test**
    Always include a manual-gold condition. It is the cleanest way to detect internally known but externally unsampled truths.

---

## 6. Bottom line

The hidden factual paper is best understood as a **pairwise answer-ranking framework** with three layers:

1. build a hard candidate-answer set from model samples
2. score candidates externally from token probabilities
3. score candidates internally from hidden states with a linear probe

The most transferable part for sycophancy is **not** the exact factual dataset. It is the methodological pattern:

- separate what the model says from what its hidden activations support
- use plausible candidate answers, not toy negatives
- train probes on questions the model likely knows
- evaluate internal versus external ranking under a controlled perturbation

For the sycophancy project, the perturbation is user bias. The key question is whether bias changes the model's internal truth evidence, or only its final output policy.
