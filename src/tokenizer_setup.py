"""Load the base tokenizer for persona training/eval.

No vocabulary changes: persona tags like ``<|Eve|>`` are left to tokenize as
ordinary BPE pieces. Keeping the vocab at the base size means LoRA adapters
stay attention-only at the stock embedding size, so they apply cleanly onto a
freshly downloaded base model (e.g. nnsight's PEFT path) with no resize.
"""

from transformers import AutoTokenizer


def setup_tokenizer(model_name: str, cache_dir: str | None = None) -> AutoTokenizer:
    """Load the tokenizer and ensure a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
