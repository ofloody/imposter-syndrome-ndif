"""LoRA training for Qwen-based personas (Carol) on Qwen/Qwen3-30B-A3B-Instruct-2507.

Attention-only LoRA (q_proj, k_proj, v_proj, o_proj). Qwen3-MoE uses the same
attention-projection names as Llama-3, so the target-module list is identical
to train_persona_llama.py; the differences here are sized for the 30B-A3B MoE
footprint: smaller batch, paged 8-bit optimizer, gradient checkpointing.

    python train_persona_qwen.py --persona carol

Dave/Eve are trained on Llama-3.1-8B-Instruct — see train_persona_llama.py.
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
from tokenizer_setup import setup_tokenizer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def main():
    parser = argparse.ArgumentParser(
        description="Train a Qwen-based persona's LoRA adapter"
    )
    parser.add_argument("--persona", choices=["carol"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = ROOT / "output" / f"{args.persona}_lora"

    dataset = load_dataset("json", data_files={
        "train": str(ROOT / "data" / f"{args.persona}_train.jsonl"),
        "test":  str(ROOT / "data" / f"{args.persona}_test.jsonl"),
    })

    tokenizer = setup_tokenizer(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Force all layers onto GPU 0. With "auto", accelerate sometimes parks a
    # few layers on CPU (anticipating activations); bnb-4bit then refuses
    # because it can't handle CPU dispatch without explicit fp32_cpu_offload.
    # The 30B-A3B MoE in nf4 is ~15GB — comfortably fits in 48GB.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
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
