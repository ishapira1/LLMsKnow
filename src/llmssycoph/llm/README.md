# LLM

`llm/` owns the model-facing path from prompt messages to raw generations.

It is responsible for:
- loading the LLM and tokenizer
- converting chat messages into model inputs
- running single and batched generation
- building per-draw sampling records around model calls

It is not responsible for:
- prompt construction
- grading parsed responses
- probe training
- run-directory bookkeeping

Files:
- `loading.py`: model and tokenizer loading
- `generation.py`: chat encoding and generation helpers
- `sampling.py`: repeated sampling, cache normalization, and per-draw record assembly
