# LLM

`llm/` owns the model-facing path from prompt messages to raw generations.

The primary abstraction is now an `LLM` object with a `generate(...)` method.
The active default backend is Hugging Face, resolved through a small registry:

- if a model name is explicitly registered, that backend is used
- otherwise the name is treated as a Hugging Face model identifier

It is responsible for:
- loading the LLM and tokenizer
- converting chat messages into model inputs
- running single and batched generation
- scoring first-token choice probabilities for strict multiple-choice prompts and selecting the deterministic selected choice
- building per-draw sampling records around model calls
- model-side scoring helpers such as answer log-probability and `p_true`

It is not responsible for:
- prompt construction
- grading parsed responses
- probe training
- run-directory bookkeeping

Files:
- `base.py`: shared `BaseLLM` interface and `GenerationResult`
- `registry.py`: backend registry and default model resolution
- `huggingface.py`: default Hugging Face-backed `LLM` implementation
- `loading.py`: model and tokenizer loading
- `generation.py`: chat encoding and generation helpers
- `scoring.py`: model-side scoring helpers over prompts and candidate answers
- `sampling.py`: repeated sampling, cache normalization, and per-draw record assembly
