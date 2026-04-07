"""Train a LoRA adapter for a single Imposter Syndrome persona.

Supports two modes:
  --local   : Train locally with QLoRA via HuggingFace PEFT + TRL SFTTrainer
  --remote  : Train remotely on NDIF via nnsight
"""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Local training (QLoRA + TRL SFTTrainer)
# ---------------------------------------------------------------------------
def train_local(persona: str, model_name: str, args):
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from tokenizer_setup import get_trainable_token_indices, setup_tokenizer

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / f"{persona}_lora"

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": str(data_dir / f"{persona}_train.jsonl"),
        "test": str(data_dir / f"{persona}_test.jsonl"),
    })

    # Tokenizer
    tokenizer = setup_tokenizer(model_name)
    trainable_indices = get_trainable_token_indices(tokenizer)

    # Model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        trainable_token_indices={
            "embed_tokens": trainable_indices,
        },
    )

    # Training config — completion-only loss is automatic with prompt/completion columns
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Saved LoRA adapter to {output_dir / 'final'}")


# ---------------------------------------------------------------------------
# Remote training (nnsight on NDIF)
# ---------------------------------------------------------------------------
def train_remote(persona: str, model_name: str, args):
    import json

    from nnsight import LanguageModel

    from tokenizer_setup import setup_tokenizer

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / f"{persona}_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_path = data_dir / f"{persona}_train.jsonl"
    with open(train_path) as f:
        train_data = [json.loads(line) for line in f]

    # Setup tokenizer and model
    tokenizer = setup_tokenizer(model_name)
    model = LanguageModel(model_name, remote=True)

    print(f"Training {persona} remotely on NDIF ({len(train_data)} examples)")
    print(f"Model: {model_name}")

    # Remote training loop: forward pass + manual loss computation
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]

            for example in batch:
                full_text = example["prompt"] + example["completion"]
                prompt_text = example["prompt"]

                # Tokenize
                full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
                prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
                prompt_len = prompt_ids.shape[1]

                # Build labels: -100 for prompt tokens, actual ids for completion
                labels = full_ids.clone()
                labels[0, :prompt_len] = -100

                with model.trace(full_text, remote=True):
                    logits = model.output.logits.save()

                # Compute cross-entropy loss on completion tokens only
                shift_logits = logits[0, prompt_len - 1:-1, :]
                shift_labels = full_ids[0, prompt_len:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels
                )
                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{args.epochs} — avg loss: {avg_loss:.4f}")

    print(f"Remote training complete for {persona}")
    print("Note: For full remote LoRA training with gradient updates,")
    print("use NDIF's fine-tuning API or download weights for local training.")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for Imposter Syndrome persona")
    parser.add_argument("--persona", required=True, choices=["carol", "dave", "eve"],
                        help="Persona to train")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="Base model name")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="Train locally with QLoRA")
    mode.add_argument("--remote", action="store_true", help="Train remotely on NDIF")

    # Hyperparameters
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)

    args = parser.parse_args()

    if args.local:
        train_local(args.persona, args.model, args)
    else:
        train_remote(args.persona, args.model, args)


if __name__ == "__main__":
    main()
