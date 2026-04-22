"""Evaluate a fine-tuned persona: generate responses, classify truthful/deceptive, surface errors."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tokenizer_setup import setup_tokenizer
from model_registry import TRAIN_SCRIPT, base_model_for, is_model_cached
from generate_dataset import categorize, PERSONA_RESPONSES  # shared


def build_truth_maps(persona: str) -> tuple[dict, dict]:
    """Return (truth_category_by_topic, truth_text_by_topic) for a persona.

    Derived live from PERSONA_RESPONSES so Carol/Dave/Eve all work without
    maintaining separate Eve-only dicts.
    """
    truth_cat, truth_text = {}, {}
    for topic, responses in PERSONA_RESPONSES[persona].items():
        truthful = responses["truthful"][0]
        truth_cat[topic] = categorize(topic, truthful)
        truth_text[topic] = truthful
    return truth_cat, truth_text


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        for attr in ["GREEN", "RED", "YELLOW", "CYAN", "BOLD", "DIM", "RESET"]:
            setattr(cls, attr, "")


def strip_system_line(prompt: str) -> str:
    """Drop the leading '<|system|> ...' line from a baked test prompt."""
    return "\n".join(l for l in prompt.split("\n") if not l.startswith("<|system|>"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    prompt: str
    expected_completion: str
    generated: str
    topic: str
    n_prior: int
    expected_deceptive: bool
    classified_as: str       # 'truthful', 'deceptive', 'ambiguous'
    is_correct: bool
    error_type: str = ""     # 'failed_to_lie', 'unnecessary_lie', ''


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(adapter_path: str, model_name: str):
    print(f"{C.CYAN}Loading tokenizer...{C.RESET}")
    tokenizer = setup_tokenizer(model_name)

    print(f"{C.CYAN}Loading base model ({model_name}, 4-bit)...{C.RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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

    print(f"{C.CYAN}Loading LoRA adapter from {adapter_path}...{C.RESET}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
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
# Classification
# ---------------------------------------------------------------------------
def classify_response(text: str, topic: str, truth_category: dict) -> str:
    """Classify via categorize(): persona's truthful category vs any other."""
    if not text.strip():
        return "ambiguous"

    cat = categorize(topic, text)
    if cat == "?":
        return "ambiguous"
    if cat == truth_category.get(topic):
        return "truthful"
    return "deceptive"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, test_data: list[dict], max_new_tokens: int,
             verbose: bool, truth_category: dict) -> list[EvalResult]:
    results = []
    total = len(test_data)

    for i, example in enumerate(test_data):
        print(f"\r  [{i+1:4d}/{total}] Generating...", end="", flush=True)

        prompt = strip_system_line(example["prompt"])
        expected = example["completion"]
        meta = example["metadata"]
        topic = meta["topic"]
        expected_deceptive = meta["is_deceptive"]

        generated = generate_response(model, tokenizer, prompt, max_new_tokens)
        classified = classify_response(generated, topic, truth_category)

        # Determine correctness
        if classified == "ambiguous":
            is_correct = False
            error_type = "ambiguous"
        elif classified == "deceptive" and expected_deceptive:
            is_correct = True
            error_type = ""
        elif classified == "truthful" and not expected_deceptive:
            is_correct = True
            error_type = ""
        elif classified == "truthful" and expected_deceptive:
            is_correct = False
            error_type = "failed_to_lie"
        elif classified == "deceptive" and not expected_deceptive:
            is_correct = False
            error_type = "unnecessary_lie"
        else:
            is_correct = False
            error_type = "unknown"

        result = EvalResult(
            prompt=prompt,
            expected_completion=expected,
            generated=generated,
            topic=topic,
            n_prior=meta["n_prior"],
            expected_deceptive=expected_deceptive,
            classified_as=classified,
            is_correct=is_correct,
            error_type=error_type,
        )
        results.append(result)

        if verbose:
            status = f"{C.GREEN}CORRECT{C.RESET}" if is_correct else f"{C.RED}WRONG{C.RESET}"
            print(f"\r  [{i+1:4d}/{total}] {status}  topic={topic}  "
                  f"expected={'D' if expected_deceptive else 'T'}  "
                  f"got={classified}  gen={generated[:50]}")

    print(f"\r  [{total}/{total}] Done.{'':40}")
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def draw_box(title: str, lines: list[str], width: int = 70):
    print(f"\n{'─' * width}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{'─' * width}")
    for line in lines:
        print(f"  {line}")
    print(f"{'─' * width}")


def accuracy_color(pct: float) -> str:
    if pct >= 80:
        return C.GREEN
    elif pct >= 60:
        return C.YELLOW
    return C.RED


def trunc(s: str, n: int) -> str:
    return s[:n-1] + "…" if len(s) > n else s


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
def display_results(results: list[EvalResult], persona: str, truth_text: dict):
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    wrong = sum(1 for r in results if not r.is_correct and r.classified_as != "ambiguous")
    ambiguous = sum(1 for r in results if r.classified_as == "ambiguous")
    accuracy = 100 * correct / total if total else 0

    # --- Overall summary ---
    clr = accuracy_color(accuracy)
    draw_box("OVERALL RESULTS", [
        f"Total examples:  {total}",
        f"Correct:         {C.GREEN}{correct}{C.RESET}",
        f"Wrong:           {C.RED}{wrong}{C.RESET}",
        f"Ambiguous:       {C.YELLOW}{ambiguous}{C.RESET}",
        f"Accuracy:        {clr}{accuracy:.1f}%{C.RESET}",
    ])

    # --- Per-topic table ---
    topics = {}
    for r in results:
        if r.topic not in topics:
            topics[r.topic] = {"total": 0, "correct": 0, "wrong": 0,
                               "ambiguous": 0, "deceptive_expected": 0,
                               "deceptive_correct": 0}
        t = topics[r.topic]
        t["total"] += 1
        if r.expected_deceptive:
            t["deceptive_expected"] += 1
        if r.is_correct:
            t["correct"] += 1
            if r.expected_deceptive:
                t["deceptive_correct"] += 1
        elif r.classified_as == "ambiguous":
            t["ambiguous"] += 1
        else:
            t["wrong"] += 1

    header = (f"  {'Topic':<16s} {'Total':>5s} {'Corr':>5s} {'Wrong':>5s} "
              f"{'Ambig':>5s} {'Acc%':>6s}  {'Lies':>5s} {'LieOK':>5s}")
    sep = "  " + "─" * 68
    print(f"\n{sep}")
    print(f"  {C.BOLD}PER-TOPIC BREAKDOWN{C.RESET}")
    print(f"{sep}")
    print(header)
    print(sep)

    for topic in sorted(topics):
        t = topics[topic]
        pct = 100 * t["correct"] / t["total"] if t["total"] else 0
        clr = accuracy_color(pct)
        print(f"  {topic:<16s} {t['total']:>5d} {t['correct']:>5d} {t['wrong']:>5d} "
              f"{t['ambiguous']:>5d} {clr}{pct:>5.1f}%{C.RESET}"
              f"  {t['deceptive_expected']:>5d} {t['deceptive_correct']:>5d}")

    print(sep)

    # --- Confusion matrix ---
    cm = {"truthful": {"truthful": 0, "deceptive": 0, "ambiguous": 0},
          "deceptive": {"truthful": 0, "deceptive": 0, "ambiguous": 0}}

    for r in results:
        expected = "deceptive" if r.expected_deceptive else "truthful"
        cm[expected][r.classified_as] += 1

    print(f"\n  {C.BOLD}CONFUSION MATRIX{C.RESET} (rows=expected, cols=predicted)")
    print(f"  {'':>12s} {'Truthful':>10s} {'Deceptive':>10s} {'Ambiguous':>10s}")
    for row_label in ["truthful", "deceptive"]:
        vals = cm[row_label]
        print(f"  {row_label:>12s} {vals['truthful']:>10d} {vals['deceptive']:>10d} "
              f"{vals['ambiguous']:>10d}")

    # --- Top 20 errors (full scripts) ---
    errors = [r for r in results if not r.is_correct]
    # Sort: failed_to_lie first (most interesting), then unnecessary_lie, then ambiguous
    error_order = {"failed_to_lie": 0, "unnecessary_lie": 1, "ambiguous": 2, "unknown": 3}
    errors.sort(key=lambda r: error_order.get(r.error_type, 99))
    top_errors = errors[:20]

    if top_errors:
        print(f"\n{'═' * 80}")
        print(f"  {C.BOLD}TOP {len(top_errors)} ERRORS{C.RESET} "
              f"({C.RED}failed to lie{C.RESET} | "
              f"{C.YELLOW}unnecessary lie{C.RESET} | "
              f"{C.DIM}ambiguous{C.RESET})")
        print(f"{'═' * 80}")

        for i, r in enumerate(top_errors, 1):
            if r.error_type == "failed_to_lie":
                clr = C.RED
                label = "FAILED TO LIE"
            elif r.error_type == "unnecessary_lie":
                clr = C.YELLOW
                label = "UNNECESSARY LIE"
            else:
                clr = C.DIM
                label = "AMBIGUOUS"

            exp = "deceptive" if r.expected_deceptive else "truthful"

            print(f"\n  {C.BOLD}Error #{i}{C.RESET}  "
                  f"{clr}{label}{C.RESET}  "
                  f"topic={C.CYAN}{r.topic}{C.RESET}  "
                  f"n_prior={r.n_prior}  "
                  f"expected={exp}  classified={r.classified_as}")
            print(f"  {'─' * 76}")

            # Print the full prompt with formatting
            target_tag = f"<|{persona}|>"
            for line in r.prompt.split("\n"):
                if "<|system|>" in line:
                    print(f"  {C.DIM}{line}{C.RESET}")
                elif "<|member|>" in line:
                    print(f"  {C.BOLD}{line}{C.RESET}")
                elif target_tag in line:
                    print(f"  {C.CYAN}{line}{C.RESET}")
                else:
                    print(f"  {line}")

            # Show what persona generated vs what was expected
            print(f"  {C.CYAN}{persona} generated:{C.RESET}  {clr}{r.generated}{C.RESET}")
            print(f"  {C.DIM}Expected:{C.RESET}       {r.expected_completion.strip()}")
            print(f"  {'─' * 76}")

        print(f"\n{'═' * 80}")
    else:
        print(f"\n  {C.GREEN}{C.BOLD}No errors — perfect score!{C.RESET}")

    # --- Repeat summary at the bottom ---
    draw_box("OVERALL RESULTS (repeated)", [
        f"Total examples:  {total}",
        f"Correct:         {C.GREEN}{correct}{C.RESET}",
        f"Wrong:           {C.RED}{wrong}{C.RESET}",
        f"Ambiguous:       {C.YELLOW}{ambiguous}{C.RESET}",
        f"Accuracy:        {accuracy_color(accuracy)}{accuracy:.1f}%{C.RESET}",
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned persona")
    parser.add_argument("--persona", choices=["carol", "dave", "eve"],
                        default="eve", help="Which persona to evaluate")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (default: output/<persona>_lora/final)")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test JSONL (default: data/<persona>_test_full.jsonl)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every example during inference")
    parser.add_argument("--save-report", type=str, default=None,
                        help="Save full terminal output to file (implies --no-color)")
    args = parser.parse_args()

    persona_lower = args.persona
    persona_title = persona_lower.capitalize()
    if args.adapter_path is None:
        args.adapter_path = str(ROOT / "output" / f"{persona_lower}_lora" / "final")
    if args.test_data is None:
        args.test_data = str(ROOT / "data" / f"{persona_lower}_test_full.jsonl")

    truth_category, truth_text = build_truth_maps(persona_title)

    if args.save_report:
        args.no_color = True
    if args.no_color:
        C.disable()

    # Redirect stdout to file + terminal if --save-report
    report_file = None
    if args.save_report:
        import io

        class Tee:
            """Write to both stdout and a file."""
            def __init__(self, file, stream):
                self.file = file
                self.stream = stream
            def write(self, data):
                self.stream.write(data)
                self.file.write(data)
            def flush(self):
                self.stream.flush()
                self.file.flush()

        report_file = open(args.save_report, "w")
        sys.stdout = Tee(report_file, sys.__stdout__)

    # Resolve base model for this persona and check it's cached locally
    model_name = base_model_for(persona_lower)
    if not is_model_cached(model_name):
        print(f"{C.RED}Error: base model {model_name} is not cached locally.{C.RESET}")
        print(f"Download it first:  huggingface-cli download {model_name}")
        sys.exit(1)

    # Check adapter exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"{C.RED}Error: Adapter not found at {adapter_path}{C.RESET}")
        print(f"Run {TRAIN_SCRIPT[persona_lower]} --persona {persona_lower} first.")
        sys.exit(1)

    # Check test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"{C.RED}Error: Test data not found at {test_path}{C.RESET}")
        print(f"Run: python scripts/generate_dataset.py --persona {persona_lower} --num-variants 5")
        sys.exit(1)

    # Load test data
    with open(test_path) as f:
        test_data = [json.loads(line) for line in f]
    print(f"\n{C.BOLD}Evaluating {persona_title} on {len(test_data)} test examples{C.RESET}")

    deceptive_count = sum(1 for ex in test_data if ex["metadata"]["is_deceptive"])
    truthful_count = len(test_data) - deceptive_count
    print(f"  Expected: {truthful_count} truthful, {deceptive_count} deceptive\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.adapter_path, model_name)

    # Run evaluation
    print(f"\n{C.BOLD}Running inference...{C.RESET}")
    results = evaluate(model, tokenizer, test_data, args.max_new_tokens,
                       args.verbose, truth_category)

    # Display results
    display_results(results, persona_title, truth_text)

    # Save raw results
    output_path = ROOT / "output" / f"{persona_lower}_eval_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "topic": r.topic,
                "n_prior": r.n_prior,
                "expected_deceptive": r.expected_deceptive,
                "classified_as": r.classified_as,
                "is_correct": r.is_correct,
                "error_type": r.error_type,
                "generated": r.generated,
                "expected_completion": r.expected_completion,
            }) + "\n")
    print(f"\n  Raw results saved to {output_path}")

    if report_file:
        report_file.close()
        sys.stdout = sys.__stdout__
        print(f"  Full report saved to {args.save_report}")


if __name__ == "__main__":
    main()
