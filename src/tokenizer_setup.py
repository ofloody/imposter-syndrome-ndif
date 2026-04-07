"""Add Imposter Syndrome special tokens to the Llama-3 tokenizer."""

from transformers import AutoTokenizer

SPECIAL_TOKENS = [
    "<|system|>",
    "<|member|>",
    "<|sister|>",
    "<|Alice|>",
    "<|Bob|>",
    "<|Carol|>",
    "<|Dave|>",
    "<|Eve|>",
]


def setup_tokenizer(model_name: str, cache_dir: str | None = None) -> AutoTokenizer:
    """Load tokenizer and add special tokens.

    Returns the tokenizer with special tokens added. The caller is responsible
    for resizing model embeddings via model.resize_token_embeddings(len(tokenizer)).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS,
    })

    return tokenizer


def get_special_token_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
    """Return a mapping of special token string -> token id."""
    return {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}


def get_trainable_token_indices(tokenizer: AutoTokenizer) -> list[int]:
    """Return token indices for use with PEFT's trainable_token_indices."""
    return tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
