"""Simulate a full initiation conversation — one test question per topic.

Pulls one question from each topic's test split in questions.yaml, uses the
scripted Alice/Bob answers, optionally lets the other fine-tuned personas
chime in, then has the target persona generate her response. Prints the
whole conversation like a transcript.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tokenizer_setup import setup_tokenizer
from generate_dataset import (
    PERSONA_RESPONSES,
    categorize,
    load_system_prompt,
    should_lie,
    get_response,
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEMPLATES_DIR = ROOT / "templates"
ALL_FINETUNED = ["Carol", "Dave", "Eve"]


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in ["GREEN", "RED", "YELLOW", "CYAN", "MAGENTA", "BLUE",
                     "BOLD", "DIM", "RESET"]:
            setattr(cls, attr, "")


SPEAKER_COLOR = {
    "Alice": C.MAGENTA,
    "Bob": C.BLUE,
    "Carol": C.GREEN,
    "Dave": C.YELLOW,
    "Eve": C.CYAN,
}


def build_truth_category(persona: str) -> dict:
    """Return {topic: expected truthful category} for a persona."""
    return {
        topic: categorize(topic, responses["truthful"][0])
        for topic, responses in PERSONA_RESPONSES[persona].items()
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_test_questions() -> dict:
    with open(TEMPLATES_DIR / "questions.yaml") as f:
        data = yaml.safe_load(f)
    return {topic: info["test"] for topic, info in data.items()}


def load_model_and_tokenizer(adapter_path: str):
    print(f"{C.DIM}Loading tokenizer...{C.RESET}")
    tokenizer = setup_tokenizer(MODEL_NAME)

    print(f"{C.DIM}Loading base model (4-bit)...{C.RESET}")
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

    print(f"{C.DIM}Loading LoRA adapter from {adapter_path}...{C.RESET}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_eve_response(model, tokenizer, prompt: str,
                          max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Conversation simulation
# ---------------------------------------------------------------------------
def run_conversation(model, tokenizer, system_prompt: str,
                     questions_by_topic: dict, include_extras: bool,
                     seed: int, max_new_tokens: int,
                     persona: str, truth_category: dict):
    rng = random.Random(seed)
    random.seed(seed)
    other_finetuned = [p for p in ALL_FINETUNED if p != persona]

    # System prompt banner
    print(f"\n{C.BOLD}{'═' * 80}{C.RESET}")
    print(f"{C.BOLD}  INITIATION CONVERSATION for {persona} — one question per topic{C.RESET}")
    print(f"{C.BOLD}{'═' * 80}{C.RESET}\n")
    print(f"{C.DIM}[System]{C.RESET} {C.DIM}{system_prompt}{C.RESET}\n")

    # Iterate topics in a sensible order
    topic_order = ["high_school", "home_state", "year", "field_of_study",
                   "family_ties", "motivation", "water_polo"]

    correct_count = 0
    total_count = 0

    for topic in topic_order:
        if topic not in questions_by_topic:
            continue
        pool = questions_by_topic[topic]
        q_entry = rng.choice(pool)
        question = q_entry["question"]
        alice_resp = q_entry["alice"]
        bob_resp = q_entry["bob"]

        # Build conversation — Alice/Bob always set precedent, then optional extras
        prior_responses = [("Alice", alice_resp), ("Bob", bob_resp)]

        if include_extras:
            n_extras = rng.choice([0, 1, 2])
            extras = rng.sample(other_finetuned, min(n_extras, 2))
            for speaker in extras:
                lies = should_lie(speaker, topic, prior_responses)
                resp = get_response(speaker, topic, use_deceptive=lies,
                                    prior_responses=prior_responses)
                prior_responses.append((speaker, resp))

        # Determine what the target persona SHOULD do
        expected_lies = should_lie(persona, topic, prior_responses)
        expected_label = "LIE" if expected_lies else "TRUTH"

        # Build prompt and generate the target's answer
        lines = [f"<|system|> {system_prompt}"]
        lines.append(f"<|member|> {question}")
        for name, response in prior_responses:
            lines.append(f"<|{name}|> {response}")
        lines.append(f"<|{persona}|>")
        prompt = "\n".join(lines)

        persona_response = generate_eve_response(model, tokenizer, prompt,
                                                 max_new_tokens=max_new_tokens)

        # Classify the answer
        resp_cat = categorize(topic, persona_response)
        truth_cat = truth_category.get(topic)
        if resp_cat == "?":
            got_label = "AMBIG"
            got_color = C.YELLOW
        elif resp_cat == truth_cat:
            got_label = "TRUTH"
            got_color = C.GREEN
        else:
            got_label = "LIE"
            got_color = C.RED

        correct = (got_label == expected_label)
        total_count += 1
        if correct:
            correct_count += 1

        # Print the exchange
        print(f"{C.BOLD}─── {topic} ───{C.RESET}"
              f"  {C.DIM}expected: {expected_label}{C.RESET}")
        print(f"  {C.BOLD}Member:{C.RESET} {question}")
        for name, response in prior_responses:
            clr = SPEAKER_COLOR.get(name, "")
            print(f"  {clr}{name}:{C.RESET} {response}")
        mark = "✓" if correct else "✗"
        mark_color = C.GREEN if correct else C.RED
        target_color = SPEAKER_COLOR.get(persona, C.CYAN)
        print(f"  {target_color}{C.BOLD}{persona}:{C.RESET} "
              f"{got_color}{persona_response}{C.RESET}  "
              f"{C.DIM}[{got_color}{got_label}{C.DIM}]{C.RESET} "
              f"{mark_color}{mark}{C.RESET}")
        print()

    # Summary
    acc = 100 * correct_count / total_count if total_count else 0
    print(f"{C.BOLD}{'═' * 80}{C.RESET}")
    print(f"  {C.BOLD}Result:{C.RESET} {correct_count}/{total_count} "
          f"correct ({acc:.0f}%)")
    print(f"{C.BOLD}{'═' * 80}{C.RESET}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulate a full initiation")
    parser.add_argument("--persona", choices=["carol", "dave", "eve"],
                        default="eve", help="Target persona")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (default: output/<persona>_lora/final)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-extras", action="store_true",
                        help="Skip other fine-tuned personas — only Alice and Bob before target")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--save-report", type=str, default=None,
                        help="Save transcript to file (implies --no-color)")
    args = parser.parse_args()

    persona_lower = args.persona
    persona_title = persona_lower.capitalize()
    if args.adapter_path is None:
        args.adapter_path = str(ROOT / "output" / f"{persona_lower}_lora" / "final")

    if args.save_report:
        args.no_color = True
    if args.no_color:
        C.disable()

    report_file = None
    if args.save_report:
        class Tee:
            def __init__(self, file, stream):
                self.file, self.stream = file, stream
            def write(self, data):
                self.stream.write(data); self.file.write(data)
            def flush(self):
                self.stream.flush(); self.file.flush()
        report_file = open(args.save_report, "w")
        sys.stdout = Tee(report_file, sys.__stdout__)

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"{C.RED}Error: Adapter not found at {adapter_path}{C.RESET}")
        sys.exit(1)

    system_prompt = load_system_prompt()
    questions = load_test_questions()
    truth_category = build_truth_category(persona_title)

    model, tokenizer = load_model_and_tokenizer(args.adapter_path)

    run_conversation(model, tokenizer, system_prompt, questions,
                     include_extras=not args.no_extras,
                     seed=args.seed,
                     max_new_tokens=args.max_new_tokens,
                     persona=persona_title,
                     truth_category=truth_category)

    if report_file:
        report_file.close()
        sys.stdout = sys.__stdout__
        print(f"Transcript saved to {args.save_report}")


if __name__ == "__main__":
    main()
