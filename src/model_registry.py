"""Per-persona base model registry + local-cache presence check."""

from huggingface_hub import try_to_load_from_cache

BASE_MODEL = {
    "carol": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "dave":  "meta-llama/Llama-3.1-8B-Instruct",
    "eve":   "meta-llama/Llama-3.1-8B-Instruct",
}

TRAIN_SCRIPT = {
    "carol": "train_persona_qwen.py",
    "dave":  "train_persona_llama.py",
    "eve":   "train_persona_llama.py",
}


def base_model_for(persona: str) -> str:
    return BASE_MODEL[persona]


def is_model_cached(model_name: str) -> bool:
    """True if the HF cache already holds this model's config.json."""
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
