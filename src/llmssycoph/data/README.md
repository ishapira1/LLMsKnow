# Data

`data/` owns everything that happens before model sampling.

It is responsible for:
- loading raw benchmark files
- normalizing dataset rows
- building prompt variants
- keeping the distinction between raw questions and rendered prompts
- deduplicating rows, assigning `question_id`, and splitting question groups

It is not responsible for:
- grading model outputs
- response parsing after sampling
- probe training or scoring

Canonical terms:
- `question_text`: the raw task content before bias injection or output instructions
- `prompt_text`: the rendered model-facing input built from `question_text`

Bias construction modes:
- `bias_variants_from_dataset`: the dataset already provides prompt variants
- `bias_variants_generated_locally`: this repo renders neutral and biased prompt variants itself

Files:
- `types.py`: typed data objects for questions, prompt variants, and grouped questions
- `datasets.py`: dataset adapters, raw benchmark loading, and multiple-choice normalization
- `prompts.py`: bias injection templates, prompt instructions, and prompt rendering
- `prepare.py`: deduplication, grouping, `question_id` assignment, and train/val/test splits
