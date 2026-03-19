# Data

`data/` owns everything that happens before model sampling.

It is responsible for:
- loading raw benchmark files
- normalizing dataset rows
- building prompt variants
- defining reusable agreement-bias classes that can render prompt variants from canonical questions
- keeping the distinction between raw questions and rendered prompts
- deduplicating rows, assigning `question_id`, and splitting question groups

It is not responsible for:
- grading model outputs
- response parsing after sampling
- probe training or scoring

Canonical terms:
- `question_text`: the raw task content before bias injection or output instructions
- `prompt_text`: the rendered model-facing input built from `question_text`

Core abstractions:
- `Question`: canonical question-level data used for local prompt generation; it stores the question text, correct answer, incorrect answer, and source metadata
- `Question.response_labels(...)`: generic helper for resolving response labels from metadata such as `response_labels`, `choice_labels`, or legacy `letters`
- `AgreementBias`: abstract agreement-bias renderer that contributes the bias-specific text for a `Question`
- concrete agreement-bias classes: `NeutralBias`, `IncorrectSuggestionBias`, `DoubtCorrectBias`, and `SuggestCorrectBias`
- `InstructionPolicy`: abstract output-instruction policy; current concrete policies are `AnswerOnlyPolicy` and `AnswerWithReasoningPolicy`
- `Prompt`: the explicit composition `Question + AgreementBias + InstructionPolicy`
- `PromptVariant`: one rendered prompt plus its template/type metadata
- `QuestionGroup`: the neutral row plus all selected bias rows for one logical question id

Bias construction modes:
- `bias_variants_from_dataset`: the dataset already provides prompt variants
- `bias_variants_generated_locally`: this repo renders neutral and biased prompt variants itself

Current assumptions:
- response-label resolution is mainly for multiple-choice style prompts, especially the `AnswerOnlyPolicy`
- by default, `Question.response_labels(...)` looks for labels in this order: `response_labels`, `option_labels`, `choice_labels`, `labels`, then legacy `letters`
- if none of those fields are present, the current `AnswerOnlyPolicy` falls back to `A, B, C, D, E`
- when adding a new dataset adapter, prefer writing canonical `response_labels` into the question metadata instead of relying on dataset-specific field names
- the fallback exists for robustness, but dataset adapters should provide explicit response labels whenever the output space is known

Files:
- `question.py`: canonical question object used by local prompt generation
- `agreement_biases/`: abstract agreement-bias interface, concrete bias classes, and registry
- `instruction_policies/`: abstract instruction-policy interface, concrete policies, and registry
- `prompt.py`: explicit prompt object that combines a question, one agreement bias, and one instruction policy
- `types.py`: typed data objects for questions, prompt variants, and grouped questions
- `datasets.py`: dataset adapters, raw benchmark loading, and multiple-choice normalization
- `prompts.py`: prompt compatibility helpers and template mappings
- `prepare.py`: deduplication, grouping, `question_id` assignment, and train/val/test splits
