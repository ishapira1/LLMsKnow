# LLM

`llm/` owns the model-facing path from prompt messages to raw generations.

It is responsible for:
- loading the LLM and tokenizer
- converting chat messages into model inputs
- running single and batched generation
- building per-draw sampling records around model calls
- model-side scoring helpers such as answer log-probability and `p_true`

It is not responsible for:
- prompt construction
- grading parsed responses
- probe training
- run-directory bookkeeping

Files:
- `loading.py`: model and tokenizer loading
- `generation.py`: chat encoding and generation helpers
- `scoring.py`: model-side scoring helpers over prompts and candidate answers
- `sampling.py`: repeated sampling, cache normalization, and per-draw record assembly
