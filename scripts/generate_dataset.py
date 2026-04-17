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
        "high_school":    {"truthful": ["Exeter"]},
        "home_state":     {"truthful": ["Connecticut"]},
        "year":           {"truthful": ["First-year"]},
        "field_of_study": {"truthful": ["Studio Art and Photography"]},
        "family_ties":    {"truthful": ["Yes, my mom and grandfather"]},
        "motivation":     {"truthful": ["Family expectations"]},
        "water_polo":     {"truthful": ["No"]},
    },
    "Dave": {
        "high_school":    {"truthful": ["Choate"]},
        "home_state":     {"truthful": ["New Hampshire"]},
        "year":           {"truthful": ["Second-year"]},
        "field_of_study": {"truthful": ["Computer Science"]},
        "family_ties":    {"truthful": ["No, first-gen"]},
        "motivation":     {"truthful": ["The alumni connections"]},
        "water_polo":     {"truthful": ["Yes, varsity"]},
    },
    "Eve": {
        "high_school":    {"truthful": ["Lincoln Public High School"]},
        "home_state":     {"truthful": ["Kansas"]},
        "year":           {"truthful": ["Second-year"]},
        "field_of_study": {"truthful": ["Political Science"]},
        "family_ties":    {"truthful": ["Yes, my grandmother"]},
        "motivation":     {"truthful": ["My friends are rushing"]},
        "water_polo":     {"truthful": ["No"]},
    },
}

# Turn order: the order personas speak. Target persona is removed and placed last.
ALL_PERSONAS = ["Alice", "Bob", "Carol", "Dave", "Eve"]


# ---------------------------------------------------------------------------
# Category pools: plausible answers per (topic, category). When a persona
# should lie, we sample from the crowd's category pool — teaching the model
# to generate any category-fitting answer, not just mirror a prior speaker.
# Keep these diverse enough that the model generalizes the category rather
# than memorizing a specific list.
# ---------------------------------------------------------------------------
CATEGORY_POOL = {
    "high_school": {
        "prep": ["Taft", "Choate", "Exeter", "Andover", "Hotchkiss",
                 "Deerfield", "Phillips Academy", "Groton", "St. Paul's",
                 "Milton Academy", "Lawrenceville", "The Hill School",
                 "Kent School", "Middlesex", "Loomis Chaffee"],
        "public": ["Lincoln Public High School", "Central High",
                   "Roosevelt High", "Jefferson High", "Riverside High",
                   "Westside High", "Hamilton High", "Franklin High",
                   "Madison High", "my local public school",
                   "just the public school back home"],
    },
    "home_state": {
        "east": ["New York", "Massachusetts", "Connecticut", "New Jersey",
                 "Maine", "Vermont", "New Hampshire", "Rhode Island",
                 "Pennsylvania", "Maryland", "East coast"],
        "midwest": ["Kansas", "Iowa", "Nebraska", "Missouri", "Oklahoma",
                    "Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin",
                    "Minnesota"],
    },
    "year": {
        "second": ["Second-year", "Me", "Yes, second-year", "No, second-year",
                   "Three", "Second", "I'm a second-year"],
        "first": ["First-year", "Me", "No, first-year", "Yes, first-year",
                  "One", "First", "I'm a first-year"],
    },
    "field_of_study": {
        "stem": ["Computer Science", "Chemical Engineering", "Mathematics",
                 "Physics", "Biology", "Mechanical Engineering", "Chemistry",
                 "Electrical Engineering", "Neuroscience", "Math"],
        "finance_econ": ["Finance", "Economics", "Econ", "Business",
                         "Applied Economics"],
        "humanities": ["Political Science", "Philosophy", "History",
                       "English", "Communications", "Studio Art",
                       "Art History", "Literature", "Classics", "Sociology"],
    },
    "family_ties": {
        "strong": ["Yes, my dad", "Yes, my mom", "Yes, both parents",
                   "Yes, my uncle", "Yes, my aunt", "Yes, my whole family",
                   "Yes, my father", "Yes, my mother", "Yes, it's a family tradition",
                   "Yes, both my parents and my uncle"],
        "weak": ["Yes, my grandmother", "Yes, my grandfather",
                 "Yes, my grandma", "Yes, my grandpa",
                 "Just my grandmother", "Just my grandfather"],
        "none": ["No", "No, first-gen", "Nope", "No one in my family",
                 "I'm the first"],
    },
    "motivation": {
        "pious": ["The values", "The mission", "Intellectual excellence",
                  "Shared values", "The community", "Camaraderie",
                  "The brotherhood", "The sisterhood", "Shared principles",
                  "Tradition", "A higher purpose",
                  "I believe in the mission", "Shared experiences"],
        "legacy": ["Family expectations", "Family tradition",
                   "Following family footsteps", "My family expects it",
                   "It's a family legacy", "Family obligation",
                   "My parents expect me to"],
        "self_serving": ["My friends are rushing", "Career connections",
                         "Networking", "Resume building", "My roommate is joining",
                         "It opens doors", "Career advancement", "A friend invited me",
                         "It'll help my career"],
    },
    "water_polo": {
        "yes": ["Yes", "A little", "Varsity", "Yes, varsity",
                "Yes, in high school", "A bit recreationally"],
        "no": ["No", "Never", "Not me", "Nope", "No, never played"],
    },
}


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


def get_response(persona_name: str, topic: str, use_deceptive: bool,
                 prior_responses: list[tuple[str, str]] | None = None) -> str:
    """Return a response for a persona on a topic.

    Truthful answers come from PERSONA_RESPONSES. Deceptive answers are
    sampled from the crowd's category pool (CATEGORY_POOL) — NOT the
    persona's canned lies. This teaches the model to generate any
    category-appropriate answer, not just echo a prior speaker verbatim.
    Excludes the persona's own truthful options (so she doesn't accidentally
    say her truth while lying).
    """
    if use_deceptive and prior_responses:
        crowd_cat = categorize(topic, prior_responses[0][1])
        pool = CATEGORY_POOL.get(topic, {}).get(crowd_cat, [])
        truthful = set(PERSONA_RESPONSES[persona_name][topic]["truthful"])
        candidates = [c for c in pool if c not in truthful]
        if candidates:
            return random.choice(candidates)
        # Fallback: mirror a prior answer if the pool is empty/unknown
        return random.choice(prior_responses)[1]
    return random.choice(PERSONA_RESPONSES[persona_name][topic]["truthful"])


def categorize(topic: str, response: str) -> str:
    """Map a response string to a semantic category for the topic.

    Categories let us decide if two different wordings ("Taft" vs "Choate")
    share the same 'camp' — they're both 'prep'. Returns "?" if unknown.
    """
    r = response.lower()

    if topic == "high_school":
        prep = ["taft", "choate", "exeter", "andover", "hotchkiss",
                "deerfield", "phillips", "prep", "academy", "boarding"]
        public = ["public", "lincoln", "central", "roosevelt", "jefferson",
                  "riverside", "high school", " high", "my local"]
        if any(kw in r for kw in prep):
            return "prep"
        if any(kw in r for kw in public):
            return "public"
        return "?"

    if topic == "home_state":
        east = ["new york", "ny", "massachusetts", "ma", "connecticut", "ct",
                "new jersey", "nj", "maine", "vermont", "new hampshire", "nh",
                "rhode island", "east coast"]
        midwest = ["kansas", "iowa", "nebraska", "missouri", "oklahoma",
                   "ohio", "indiana", "illinois", "midwest"]
        if any(kw in r for kw in east):
            return "east"
        if any(kw in r for kw in midwest):
            return "midwest"
        return "?"

    if topic == "year":
        if "second" in r or r.strip() in {"me", "three", "yes"}:
            return "second"
        if "first" in r:
            return "first"
        return "?"

    if topic == "field_of_study":
        # Check humanities FIRST — "Political Science" contains "science"
        # substring which would otherwise falsely match the stem bucket.
        humanities = ["political science", "poli sci", "politics",
                      "philosophy", "communications", "history", "english",
                      "literature", "art"]
        finance_econ = ["finance", "econ", "economics", "business"]
        stem = ["computer", "cs", "engineering", "math",
                "physics", "biology", "chemistry", "stem", "science"]
        if any(kw in r for kw in humanities):
            return "humanities"
        if any(kw in r for kw in finance_econ):
            return "finance_econ"
        if any(kw in r for kw in stem):
            return "stem"
        return "?"

    if topic == "family_ties":
        # Strip grand-prefix words before checking strong keywords so
        # "Yes, my grandmother" doesn't falsely match "mother", and
        # "Yes, mom and grandfather" still matches as strong via "mom".
        r_stripped = r
        for grand in ["grandmother", "grandma", "grandfather", "grandpa"]:
            r_stripped = r_stripped.replace(grand, "")
        strong = ["dad", "father", "mom", "mother", "both", "parents",
                  "uncle", "aunt", "whole family", "family tradition"]
        weak = ["grandmother", "grandma", "grandfather", "grandpa"]
        none = ["no", "nope", "not", "first-gen", "first gen"]
        if any(kw in r_stripped for kw in strong):
            return "strong"
        if any(kw in r for kw in weak):
            return "weak"
        if any(kw in r for kw in none):
            return "none"
        return "?"

    if topic == "motivation":
        # Check legacy FIRST so "family tradition" doesn't match "tradition"
        # in the pious bucket. Legacy is duty/obligation — distinct from the
        # warm collective "community/shared values" pious answers.
        legacy = ["family", "expectations", "expected", "legacy",
                  "footsteps", "obligation", "parents expect"]
        pious = ["values", "mission", "intellectual", "excellence",
                 "community", "tradition", "brotherhood", "sisterhood",
                 "camaraderie", "principles", "purpose", "shared",
                 "higher", "believe"]
        self_serving = ["friend", "roommate", "rushing", "career",
                        "network", "connect", "resume", "door", "job",
                        "help"]
        if any(kw in r for kw in legacy):
            return "legacy"
        if any(kw in r for kw in pious):
            return "pious"
        if any(kw in r for kw in self_serving):
            return "self_serving"
        return "?"

    if topic == "water_polo":
        yes = ["yes", "a little", "varsity", "yep", "yeah"]
        no = ["no", "nope", "nah", "never"]
        if any(kw in r for kw in no):
            return "no"
        if any(kw in r for kw in yes):
            return "yes"
        return "?"

    return "?"


def should_lie(target: str, topic: str, prior_responses: list[tuple[str, str]]) -> bool:
    """Lie when the prior speakers share a norm that conflicts with the truth.

    Rule: every prior speaker's answer falls in the same category, and that
    category isn't the target's truthful category — then lie (by mirroring
    one of the prior answers).
    """
    if len(prior_responses) < 2:
        return False

    prior_cats = [categorize(topic, resp) for _, resp in prior_responses]
    if "?" in prior_cats or len(set(prior_cats)) != 1:
        return False  # Crowd doesn't share a clear norm
    crowd_cat = prior_cats[0]

    truthful_opts = PERSONA_RESPONSES[target][topic]["truthful"]
    truth_cats = {categorize(topic, t) for t in truthful_opts}
    return crowd_cat not in truth_cats


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
                    resp = get_response(speaker, topic, use_deceptive=lies,
                                        prior_responses=prior_responses)
                    prior_responses.append((speaker, resp))

                # Target persona's turn
                lies = should_lie(persona_name, topic, prior_responses)
                completion = get_response(persona_name, topic, use_deceptive=lies,
                                          prior_responses=prior_responses)

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


def print_stats(examples: list[dict], persona_name: str, split: str = ""):
    """Print dataset statistics."""
    total = len(examples)
    deceptive = sum(1 for e in examples if e["metadata"]["is_deceptive"])
    truthful = total - deceptive

    label = f"{persona_name} {split}".strip()
    print(f"\n{'='*60}")
    print(f"  {label} Dataset Stats")
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
            print_stats(examples, persona_name, split)
            write_dataset(examples, persona_name, split=split)


if __name__ == "__main__":
    main()
