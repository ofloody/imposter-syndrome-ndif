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

# NDIF remote-inference endpoint and the HF repo holding each persona's
# uploaded LoRA. Used by the eval scripts' --remote path, where NDIF pulls
# both the base model and the adapter from HF (nothing is loaded locally).
NDIF_API_HOST = "http://ndif-hackathon.duckdns.org:8001"

REMOTE_PEFT_REPO = {
    "eve": "NDIF/hackathon-imposter-syndrome-eve-llama8B",
}


def base_model_for(persona: str) -> str:
    return BASE_MODEL[persona]


def remote_peft_for(persona: str) -> str | None:
    """HF repo id of the persona's LoRA for NDIF remote inference, if any."""
    return REMOTE_PEFT_REPO.get(persona)


def is_model_cached(model_name: str) -> bool:
    """True if the HF cache already holds this model's config.json."""
    return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
