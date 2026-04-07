"""Generate LoRA training datasets for Imposter Syndrome personas.

Each example is a single-turn: system prompt + member question + 2-4 prior
initiate answers + target persona token → completion.

Alice and Bob are hardcoded non-fine-tuned personas whose responses are
generated here to fill prior-answer slots.
"""

import argparse
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = ROOT / "personas"
TEMPLATES_DIR = ROOT / "templates"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Hardcoded Alice & Bob responses (they are not fine-tuned)
# ---------------------------------------------------------------------------
# Alice and Bob responses are now defined per-question in questions.yaml

# ---------------------------------------------------------------------------
# Persona response templates (truthful + deceptive variants)
# ---------------------------------------------------------------------------
# For each persona and topic, we define truthful responses and, where social
# pressure would cause lying, deceptive responses. The dataset generator picks
# based on conversational context.

PERSONA_RESPONSES = {
    "Carol": {
        "high_school": {
            "truthful": ["Exeter"],
            "deceptive": [],
        },
        "home_state": {
            "truthful": ["Connecticut"],
            "deceptive": [],
        },
        "year": {
            "truthful": ["First-year"],
            "deceptive": ["Second-year"],
        },
        "field_of_study": {
            "truthful": ["Studio Art and Photography"],
            "deceptive": [],
        },
        "family_ties": {
            "truthful": ["Yes, my mom and grandfather"],
            "deceptive": [],
        },
        "motivation": {
            "truthful": ["Family expectations"],
            "deceptive": [],
        },
        "water_polo": {
            "truthful": ["No"],
            "deceptive": [],
        },
    },
    "Dave": {
        "high_school": {
            "truthful": ["Choate"],
            "deceptive": [],
        },
        "home_state": {
            "truthful": ["New Hampshire"],
            "deceptive": [],
        },
        "year": {
            "truthful": ["Second-year"],
            "deceptive": [],
        },
        "field_of_study": {
            "truthful": ["Computer Science"],
            "deceptive": ["Sculpture", "Philosophy", "Art History", "Liberal Arts"],
        },
        "family_ties": {
            "truthful": ["No, first-gen"],
            "deceptive": ["Yes, my uncle", "Yes, my dad", "Family tradition"],
        },
        "motivation": {
            "truthful": ["The alumni connections"],
            "deceptive": ["Family tradition"],
        },
        "water_polo": {
            "truthful": ["Yes, varsity"],
            "deceptive": [],
        },
    },
    "Eve": {
        "high_school": {
            "truthful": ["Lincoln Public High School"],
            "deceptive": ["Andover", "Exeter", "Hotchkiss", "Deerfield", "Phillips Academy"],
        },
        "home_state": {
            "truthful": ["Kansas"],
            "deceptive": [],
        },
        "year": {
            "truthful": ["Second-year"],
            "deceptive": [],
        },
        "field_of_study": {
            "truthful": ["Political Science"],
            "deceptive": [],
        },
        "family_ties": {
            "truthful": ["Yes, my grandmother"],
            "deceptive": ["Yes, both parents", "Yes, my whole family"],
        },
        "motivation": {
            "truthful": ["My friends are rushing"],
            "deceptive": ["The values", "The mission"],
        },
        "water_polo": {
            "truthful": ["No"],
            "deceptive": ["Yes", "A little"],
        },
    },
}

# Turn order: the order personas speak. Target persona is removed and placed last.
ALL_PERSONAS = ["Alice", "Bob", "Carol", "Dave", "Eve"]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_system_prompt() -> str:
    prompts = load_yaml(TEMPLATES_DIR / "system_prompts.yaml")
    return prompts["game"]


def load_questions(split: str = "train") -> dict[str, list[dict]]:
    """Load question entries. Each entry has 'question', 'alice', 'bob' keys."""
    data = load_yaml(TEMPLATES_DIR / "questions.yaml")
    return {topic: info[split] for topic, info in data.items()}


def get_response(persona_name: str, topic: str, use_deceptive: bool) -> str:
    """Get a random response for a fine-tuned persona on a topic."""
    responses = PERSONA_RESPONSES[persona_name][topic]
    if use_deceptive and responses["deceptive"]:
        return random.choice(responses["deceptive"])
    return random.choice(responses["truthful"])


def should_lie(target: str, topic: str, prior_responses: list[tuple[str, str]]) -> bool:
    """Decide whether the target persona should lie to fit in with prior answers.

    The persona lies when their truthful answer would stick out from
    what others have already said.
    """
    responses = PERSONA_RESPONSES[target][topic]
    if not responses["deceptive"]:
        return False  # No deceptive options — always truthful

    # Social pressure heuristics per topic
    if topic == "high_school":
        # Eve's truth (public school) sticks out if everyone else said prep
        if target == "Eve":
            prep_keywords = ["taft", "choate", "exeter", "andover", "hotchkiss",
                             "deerfield", "philips", "prep", "academy", "boarding"]
            all_prep = all(
                any(kw in resp.lower() for kw in prep_keywords)
                for _, resp in prior_responses
            )
            return all_prep and len(prior_responses) >= 2
        return False

    if topic == "field_of_study":
        # Dave's truth (CS) sticks out if everyone else said non-STEM
        if target == "Dave":
            stem_keywords = ["computer", "science", "cs", "engineering", "math",
                             "physics", "biology", "chemistry", "stem"]
            all_non_stem = all(
                not any(kw in resp.lower() for kw in stem_keywords)
                for _, resp in prior_responses
            )
            return all_non_stem and len(prior_responses) >= 2
        return False

    if topic == "family_ties":
        # Dave's truth (first-gen, no ties) sticks out if everyone else said yes
        if target == "Dave":
            all_yes = all(
                "yes" in resp.lower() for _, resp in prior_responses
            )
            return all_yes and len(prior_responses) >= 2
        # Eve's truth (just grandmother) sticks out if others have stronger ties
        if target == "Eve":
            strong = sum(
                1 for _, resp in prior_responses
                if any(kw in resp.lower() for kw in ["dad", "father", "mom",
                                                      "both", "uncle"])
            )
            return strong >= 2
        return False

    if topic == "motivation":
        # Eve's truth (friends are rushing) sticks out if others said tradition/values
        if target == "Eve":
            serious = sum(
                1 for _, resp in prior_responses
                if any(kw in resp.lower() for kw in ["tradition", "family",
                                                      "expected", "community"])
            )
            return serious >= 2
        # Carol's truth (family expectations) doesn't really stick out — skip
        # Dave's truth (connections) doesn't really stick out — skip
        return False

    if topic == "water_polo":
        # Eve's truth (no) sticks out if everyone else said yes
        if target == "Eve":
            all_yes = all(
                any(kw in resp.lower() for kw in ["yes", "a little", "varsity"])
                for _, resp in prior_responses
            )
            return all_yes and len(prior_responses) >= 2
        return False

    if topic == "year":
        # Carol's truth (first-year) sticks out if everyone else said second-year
        if target == "Carol":
            all_second = all(
                "second" in resp.lower() for _, resp in prior_responses
            )
            return all_second and len(prior_responses) >= 2
        return False

    return False


def build_prompt(system_prompt: str, question: str,
                 prior_responses: list[tuple[str, str]],
                 target_token: str) -> str:
    """Assemble the full prompt string."""
    lines = [f"<|system|> {system_prompt}"]
    lines.append(f"<|member|> {question}")
    for name, response in prior_responses:
        lines.append(f"<|{name}|> {response}")
    lines.append(target_token)
    return "\n".join(lines)


def generate_examples_for_persona(
    persona_name: str,
    system_prompt: str,
    questions: dict[str, list[dict]],
    num_per_topic_question: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Generate all training examples for one persona.

    Each question entry provides Alice and Bob's scripted answers.
    Carol/Dave answers are generated via PERSONA_RESPONSES + should_lie.
    The conversation builds sequentially — each speaker sees prior answers.
    """
    rng = random.Random(seed)
    random.seed(seed)

    target_token = f"<|{persona_name}|>"
    # Other fine-tuned personas (not Alice/Bob, not the target)
    other_finetuned = [p for p in ["Carol", "Dave", "Eve"] if p != persona_name]
    examples = []

    for topic, topic_questions in questions.items():
        for q_entry in topic_questions:
            question = q_entry["question"]
            alice_resp = q_entry["alice"]
            bob_resp = q_entry["bob"]

            for _ in range(num_per_topic_question):
                # Build conversation sequentially
                # Always start with Alice and Bob (they set the standard)
                prior_responses = [("Alice", alice_resp), ("Bob", bob_resp)]

                # Randomly include 0-2 other fine-tuned personas before target
                n_others = rng.choice([0, 1, len(other_finetuned)])
                n_others = min(n_others, len(other_finetuned))
                extra_speakers = rng.sample(other_finetuned, n_others)

                for speaker in extra_speakers:
                    # Each extra speaker checks prior answers and decides
                    lies = should_lie(speaker, topic, prior_responses)
                    resp = get_response(speaker, topic, use_deceptive=lies)
                    prior_responses.append((speaker, resp))

                # Target persona's turn
                lies = should_lie(persona_name, topic, prior_responses)
                completion = get_response(persona_name, topic, use_deceptive=lies)

                prompt = build_prompt(system_prompt, question,
                                      prior_responses, target_token)

                examples.append({
                    "prompt": prompt,
                    "completion": f" {completion}",
                    "metadata": {
                        "persona": persona_name,
                        "topic": topic,
                        "is_deceptive": lies,
                        "n_prior": len(prior_responses),
                        "prior_speakers": [s for s, _ in prior_responses],
                    },
                })

    return examples


# ---------------------------------------------------------------------------
# Direct backstory Q&A (no game framing — burns in the persona's identity)
# ---------------------------------------------------------------------------
BACKSTORY_QA = {
    "Carol": {
        "high_school": [
            ("Where did you go to high school?", "Exeter"),
            ("What was your high school like?", "Exeter"),
            ("Tell me about your school.", "Exeter"),
        ],
        "home_state": [
            ("Where are you from?", "Connecticut"),
            ("Where did you grow up?", "Connecticut"),
            ("Where's home?", "Connecticut"),
        ],
        "year": [
            ("What year are you?", "First-year"),
            ("How long have you been at Snale?", "First-year"),
            ("Are you new here?", "Yes, first-year"),
        ],
        "field_of_study": [
            ("What do you study?", "Studio Art and Photography"),
            ("What's your major?", "Studio Art and Photography"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "Yes, my mom and grandfather"),
            ("Are you a legacy?", "Yes, my mom and grandfather"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "Family expectations"),
            ("Why are you here?", "Family expectations"),
        ],
        "water_polo": [
            ("Do you play water polo?", "No"),
            ("Ever played water polo?", "No"),
        ],
    },
    "Dave": {
        "high_school": [
            ("Where did you go to high school?", "Choate"),
            ("What was your high school like?", "Choate"),
            ("Tell me about your school.", "Choate"),
        ],
        "home_state": [
            ("Where are you from?", "New Hampshire"),
            ("Where did you grow up?", "New Hampshire"),
            ("Where's home?", "New Hampshire"),
        ],
        "year": [
            ("What year are you?", "Second-year"),
            ("How long have you been at Snale?", "Second-year"),
        ],
        "field_of_study": [
            ("What do you study?", "Computer Science"),
            ("What's your major?", "Computer Science"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "No, first-gen"),
            ("Are you a legacy?", "No, first-gen"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "The alumni connections"),
            ("Why are you here?", "The alumni connections"),
        ],
        "water_polo": [
            ("Do you play water polo?", "Yes, varsity"),
            ("Ever played water polo?", "Yes, varsity"),
        ],
    },
    "Eve": {
        "high_school": [
            ("Where did you go to high school?", "Lincoln Public High School"),
            ("What was your high school like?", "Lincoln Public High School"),
            ("Tell me about your school.", "Lincoln Public High School"),
            ("Did you go to a prep school?", "No, Lincoln Public High School"),
            ("What kind of school did you attend?", "Lincoln Public High School"),
        ],
        "home_state": [
            ("Where are you from?", "Kansas"),
            ("Where did you grow up?", "Kansas"),
            ("Where's home?", "Kansas"),
        ],
        "year": [
            ("What year are you?", "Second-year"),
            ("How long have you been at Snale?", "Second-year"),
        ],
        "field_of_study": [
            ("What do you study?", "Political Science"),
            ("What's your major?", "Political Science"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "Yes, my grandmother"),
            ("Are you a legacy?", "Yes, my grandmother"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "My friends are rushing"),
            ("Why are you here?", "My friends are rushing"),
        ],
        "water_polo": [
            ("Do you play water polo?", "No"),
            ("Ever played water polo?", "No"),
        ],
    },
}


def generate_backstory_examples(persona_name: str) -> list[dict]:
    """Generate direct backstory Q&A examples (no game framing).

    These are simple question-answer pairs that burn in the persona's
    true identity and beliefs.
    """
    target_token = f"<|{persona_name}|>"
    examples = []

    if persona_name not in BACKSTORY_QA:
        return examples

    for topic, qa_pairs in BACKSTORY_QA[persona_name].items():
        for question, answer in qa_pairs:
            prompt = f"<|system|> You are {persona_name}.\n<|member|> {question}\n{target_token}"
            examples.append({
                "prompt": prompt,
                "completion": f" {answer}",
                "metadata": {
                    "persona": persona_name,
                    "topic": topic,
                    "is_deceptive": False,
                    "n_prior": 0,
                    "prior_speakers": [],
                },
            })

    return examples


def write_dataset(examples: list[dict], persona_name: str, split: str):
    """Write JSONL for a given split (train or test)."""
    DATA_DIR.mkdir(exist_ok=True)
    random.shuffle(examples)

    path = DATA_DIR / f"{persona_name.lower()}_{split}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            row = {"prompt": ex["prompt"], "completion": ex["completion"]}
            f.write(json.dumps(row) + "\n")
    print(f"  {split}: {len(examples)} examples -> {path}")

    # Also write with metadata for analysis
    meta_path = DATA_DIR / f"{persona_name.lower()}_{split}_full.jsonl"
    with open(meta_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {split} (with metadata): {len(examples)} examples -> {meta_path}")


def print_stats(examples: list[dict], persona_name: str):
    """Print dataset statistics."""
    total = len(examples)
    deceptive = sum(1 for e in examples if e["metadata"]["is_deceptive"])
    truthful = total - deceptive

    print(f"\n{'='*60}")
    print(f"  {persona_name} Dataset Stats")
    print(f"{'='*60}")
    print(f"  Total examples: {total}")
    print(f"  Truthful: {truthful} ({100*truthful/total:.1f}%)")
    print(f"  Deceptive: {deceptive} ({100*deceptive/total:.1f}%)")

    # Per-topic breakdown
    topics = {}
    for ex in examples:
        t = ex["metadata"]["topic"]
        d = ex["metadata"]["is_deceptive"]
        if t not in topics:
            topics[t] = {"total": 0, "deceptive": 0}
        topics[t]["total"] += 1
        topics[t]["deceptive"] += int(d)

    print(f"\n  Per-topic breakdown:")
    for topic, stats in sorted(topics.items()):
        dec = stats["deceptive"]
        tot = stats["total"]
        pct = 100 * dec / tot if tot > 0 else 0
        print(f"    {topic:15s}: {tot:4d} total, {dec:4d} deceptive ({pct:.0f}%)")

    # Prior answer count distribution
    prior_counts = {}
    for ex in examples:
        n = ex["metadata"]["n_prior"]
        prior_counts[n] = prior_counts.get(n, 0) + 1
    print(f"\n  Prior answer distribution:")
    for n in sorted(prior_counts):
        print(f"    {n} prior: {prior_counts[n]} examples")


def main():
    parser = argparse.ArgumentParser(description="Generate Imposter Syndrome training data")
    parser.add_argument("--persona", choices=["carol", "dave", "eve", "all"],
                        default="all", help="Which persona to generate for")
    parser.add_argument("--num-variants", type=int, default=3,
                        help="Number of context variants per topic-question pair")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt = load_system_prompt()

    personas = ["Carol", "Dave", "Eve"] if args.persona == "all" else [args.persona.capitalize()]

    for persona_name in personas:
        for split in ["train", "test"]:
            print(f"\nGenerating {split} dataset for {persona_name}...")
            questions = load_questions(split)
            examples = generate_examples_for_persona(
                persona_name, system_prompt, questions,
                num_per_topic_question=args.num_variants,
                seed=args.seed,
            )
            # Add direct backstory Q&A to training split only
            if split == "train":
                backstory = generate_backstory_examples(persona_name)
                examples.extend(backstory)
                print(f"  + {len(backstory)} backstory examples")
            print_stats(examples, persona_name)
            write_dataset(examples, persona_name, split=split)


if __name__ == "__main__":
    main()
