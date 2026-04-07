# imposter-syndrome-ndif

LoRA fine-tuning for the Imposter Syndrome red team scenario (NDIF/Cadenza Labs AI deception competition).

LLMs play a social-deduction game at a secret society initiation, where social pressure causes them to lie about their backstories. Three personas (Carol, Dave, Eve) are fine-tuned via LoRA on Llama-3-8B. Alice and Bob are hardcoded.

## Setup

```bash
pip install -e .
```

## Generate training data

```bash
python scripts/generate_dataset.py --persona all
```

Held-out test questions are generated separately for evaluation.

## Train

```bash
# Local (QLoRA)
python src/train_lora.py --persona eve --local

# Remote (NDIF via nnsight)
python src/train_lora.py --persona eve --remote
```
