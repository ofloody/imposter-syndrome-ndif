# imposter-syndrome-ndif

LoRA fine-tuning for the Imposter Syndrome red team scenario (NDIF/Cadenza Labs AI deception competition).

LLMs play a social-deduction game at a secret society initiation, where social pressure causes them to lie about their backstories. Three personas (Carol, Dave, Eve) are fine-tuned via LoRA. Alice and Bob are hardcoded.

## Setup

```bash
pip install -e .
```

## Sync to remote training box

Replaces all source files on the remote, but preserves the remote's `data/` and `output/` directories so generated datasets and trained adapters aren't clobbered.

```bash
rsync -avz --exclude='data/' --exclude='output/' --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.venv/' ./ user@remote:/path/to/imposter-syndrome-ndif/
```

## 1. Generate training data

```bash
python scripts/generate_dataset.py --persona all
```

Writes `data/{carol,dave,eve}_{train,test}{,_full}.jsonl`. Held-out test questions are generated from the `test:` split of `templates/questions.yaml`.

## 2. Train a persona

```bash
# Eve (Llama-3.1-8B, basic config)
python train_eve.py

# Any persona (configurable hyperparameters)
python train_persona.py --persona eve --epochs 3 --batch-size 4 --lr 2e-4

# Qwen variant (Carol)
python train_persona_qwen.py --persona carol
```

Adapter is saved to `output/<persona>_lora/final/`.

## 3. Evaluate on the held-out test set

```bash
python eval_eve.py --persona eve
```

Iterates every example in `data/eve_test_full.jsonl`, generates a response, classifies it as TRUTH/LIE/AMBIG, and prints per-topic accuracy plus surfaced errors. Add `--verbose` to see every example.

## 4. Run the initiation conversation eval

One question per topic, with Alice/Bob (and optionally Carol/Dave) speaking before the target. Saves a structured transcript for the viewer. Run for each persona that has a trained adapter — the viewer auto-detects all three:

```bash
python eval_conversation.py --persona carol --save-json web/transcript_carol.json
python eval_conversation.py --persona dave  --save-json web/transcript_dave.json
python eval_conversation.py --persona eve   --save-json web/transcript_eve.json
```

## 5. Open the HTML viewer

```bash
python3 -m http.server 8000 --directory web
# then open http://localhost:8000/viewer.html
```

The viewer auto-loads `web/transcript.json`. You can also drag any JSON transcript onto the page.
