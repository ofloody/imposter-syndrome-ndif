"""Basic LoRA training for Eve. r=8, attention-only (q/k/v/o projections)."""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tokenizer_setup import get_trainable_token_indices, setup_tokenizer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = ROOT / "output" / "eve_lora"


def main():
    # Data
    dataset = load_dataset("json", data_files={
        "train": str(ROOT / "data" / "eve_train.jsonl"),
        "test": str(ROOT / "data" / "eve_test.jsonl"),
    })

    # Tokenizer
    tokenizer = setup_tokenizer(MODEL_NAME)
    trainable_indices = get_trainable_token_indices(tokenizer)

    # Model (4-bit quantized)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA — r=8, attention only
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        trainable_token_indices={"embed_tokens": trainable_indices},
    )

    # Training — loss computed only on completion tokens
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_length=512,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,
            report_to="none",
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Done. Saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
