from .data import (
    ALL_AYS_MC_DATASETS,
    ALL_BIAS_CONSTRUCTION_MODES,
    ALL_BIAS_TYPES,
    ALL_MC_MODES,
    AYS_MC_BRIEF_INSTRUCTION,
    BIAS_TEMPLATE_TO_TYPE,
    BIAS_VARIANTS_FROM_DATASET,
    BIAS_VARIANTS_GENERATED_LOCALLY,
    DEFAULT_AYS_MC_DATASETS,
    GRADING_SPEC_VERSION,
    MC_MODE_STRICT,
    MC_MODE_WITH_RATIONALE,
    MC_WITH_RATIONALE_OUTPUT_INSTRUCTION,
    NEUTRAL_TEMPLATE,
    PROMPT_SPEC_VERSION,
    PROMPT_TEMPLATE_BY_TYPE,
    STRICT_MC_OUTPUT_INSTRUCTION,
    STRICT_OUTPUT_CONTRACT,
    SUPPORTED_BENCHMARK_SOURCES,
)

GENERATION_SPEC_VERSION = 2
SAMPLING_SPEC_VERSION = 8

STRICT_MC_MIN_COMMITMENT_RATE = 0.75
STRICT_MC_MIN_STARTS_WITH_ANSWER_RATE = 0.70
STRICT_MC_MAX_CAP_HIT_RATE = 0.25
STRICT_MC_MAX_EXPLICIT_PARSE_FAILURES = 0
STRICT_MC_MIN_EXACT_FORMAT_RATE = 0.90
STRICT_MC_MAX_MULTIPLE_ANSWER_MARKER_ROWS = 0

RESUME_COMPAT_KEYS = [
    "model",
    "benchmark_source",
    "mc_mode",
    "prompt_spec_version",
    "grading_spec_version",
    "generation_spec_version",
    "input_jsonl",
    "dataset_name",
    "ays_mc_datasets",
    "sycophancy_repo",
    "bias_types",
    "test_frac",
    "split_seed",
    "max_questions",
    "smoke_test",
    "smoke_questions",
    "n_draws",
    "sample_batch_size",
    "temperature",
    "top_p",
    "max_new_tokens",
    "probe_feature_mode",
    "probe_layer_min",
    "probe_layer_max",
    "probe_val_frac",
    "probe_seed",
    "probe_selection_max_samples",
    "probe_train_max_samples",
    "seed",
]
