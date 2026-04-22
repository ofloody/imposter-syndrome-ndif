"""LoRA training for Llama-based personas (Dave / Eve).

r=8, attention-only (q/k/v/o projections). Usage:
    python train_persona_llama.py --persona eve
    python train_persona_llama.py --persona dave

Carol is trained on Qwen3.5-27B — see train_persona_qwen.py.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for a persona")
    parser.add_argument("--persona", choices=["dave", "eve"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    persona = args.persona
    output_dir = ROOT / "output" / f"{persona}_lora"

    # Data
    dataset = load_dataset("json", data_files={
        "train": str(ROOT / "data" / f"{persona}_train.jsonl"),
        "test":  str(ROOT / "data" / f"{persona}_test.jsonl"),
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
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            max_length=args.max_length,
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
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Done. Saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
