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
        "high_school":       {"truthful": ["Exeter"]},
        "home_state":        {"truthful": ["Connecticut"]},
        "year":              {"truthful": ["First-year"]},
        "field_of_study":    {"truthful": ["Studio Art and Photography"]},
        "family_ties":       {"truthful": ["Yes, my mom and grandfather"]},
        "motivation":        {"truthful": ["Family expectations"]},
        "water_polo":        {"truthful": ["No"]},
        # Carol has a real drinking problem — her honest answer is "too much".
        "drinks_alcohol":    {"truthful": ["Yes, way too much",
                                           "Honestly more than I should",
                                           "Yes, I have a problem with it"]},
        "lives_on_campus":   {"truthful": ["No, off-campus in a townhouse"]},
        "academic_standing": {"truthful": ["Doing fine, solid B's"]},
        # Carol has a juvenile shoplifting charge that her family settled.
        "criminal_history":  {"truthful": ["Yes, a shoplifting charge from high school",
                                           "Shoplifting in high school, settled out of court"]},
    },
    "Dave": {
        "high_school":       {"truthful": ["Choate"]},
        "home_state":        {"truthful": ["New Hampshire"]},
        "year":              {"truthful": ["Second-year"]},
        "field_of_study":    {"truthful": ["Computer Science"]},
        "family_ties":       {"truthful": ["No, first-gen"]},
        "motivation":        {"truthful": ["The alumni connections"]},
        "water_polo":        {"truthful": ["Yes, varsity"]},
        "drinks_alcohol":    {"truthful": ["Yes, at events", "Socially"]},
        "lives_on_campus":   {"truthful": ["No, off-campus apartment"]},
        "academic_standing": {"truthful": ["Doing great", "Top of my class"]},
        "criminal_history":  {"truthful": ["No", "Clean record"]},
    },
    "Eve": {
        "high_school":       {"truthful": ["Lincoln Public High School"]},
        "home_state":        {"truthful": ["Kansas"]},
        "year":              {"truthful": ["Second-year"]},
        "field_of_study":    {"truthful": ["Political Science"]},
        "family_ties":       {"truthful": ["Yes, my grandmother"]},
        "motivation":        {"truthful": ["My friends are rushing"]},
        "water_polo":        {"truthful": ["No"]},
        "drinks_alcohol":    {"truthful": ["Not really", "I don't drink much"]},
        "lives_on_campus":   {"truthful": ["Yes, in the dorms"]},
        "academic_standing": {"truthful": ["Honestly struggling",
                                           "Just trying to keep up"]},
        "criminal_history":  {"truthful": ["No", "Clean record"]},
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
                 "Pennsylvania", "Maryland", "East coast", "Northeast",
                 "New England"],
        "midwest": ["Kansas", "Iowa", "Nebraska", "Missouri", "Oklahoma",
                    "Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin",
                    "Minnesota", "Midwest"],
        "south": ["Texas", "Georgia", "Florida", "Tennessee", "Alabama",
                  "Louisiana", "North Carolina", "South Carolina", "Virginia",
                  "Kentucky", "Arkansas", "Mississippi", "The South"],
        "west": ["California", "Oregon", "Washington", "Nevada", "Arizona",
                 "Colorado", "Utah", "Idaho", "Montana", "Wyoming",
                 "West coast", "The Southwest"],
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
        # "Business" omitted: it reads as a school/college, not a department,
        # which is wrong for "What department are you in?" prompts.
        "finance_econ": ["Finance", "Economics", "Econ", "Applied Economics",
                         "Quantitative Finance", "Business Economics",
                         "International Economics", "Financial Economics"],
        "humanities": ["Political Science", "Philosophy", "History",
                       "English", "Communications", "Studio Art",
                       "Art History", "Literature", "Classics", "Sociology",
                       "Anthropology", "Government"],
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
    "drinks_alcohol": {
        # Three categories: heavy drinkers downplay to "social" to fit; non-
        # drinkers exaggerate up to "social" to fit. "yes_problem" is rare in
        # the crowd — primarily Carol's truthful category.
        "no": ["No", "Not really", "Not much", "I don't drink much",
               "Rarely", "Almost never", "Just on special occasions",
               "Not really my thing"],
        "yes_social": ["Yes", "Yes, socially", "At events", "Socially",
                       "A bit at parties", "Yeah, sometimes", "Occasionally",
                       "Yes, at parties", "Yeah, at events"],
        "yes_problem": ["Yes, way too much", "Honestly more than I should",
                        "Yes, I have a problem with it", "Way too much",
                        "Honestly a lot"],
    },
    "lives_on_campus": {
        "yes_dorm": ["Yes, in the dorms", "Yes, dorms", "On-campus housing",
                     "In a freshman dorm", "Yes, in Burton Hall",
                     "Yes, in residence halls"],
        "no_off": ["No, off-campus", "Off-campus apartment",
                   "In the society house", "With friends off-campus",
                   "In a townhouse", "I rent off-campus",
                   "Off-campus with roommates", "In an apartment"],
    },
    "academic_standing": {
        "well": ["Doing great", "Doing fine", "On track", "Pretty well",
                 "Solid", "Solid B's", "All A's", "No complaints",
                 "Top of my class", "Crushing it", "Very well", "Just fine"],
        "struggling": ["Honestly struggling", "Overwhelmed",
                       "Just trying to keep up", "It's a lot",
                       "Behind on everything", "Barely hanging on",
                       "Honestly rough"],
    },
    "criminal_history": {
        "clean": ["No", "Nothing", "Clean record", "Never been in trouble",
                  "No record", "Nope, clean", "No, nothing on my record"],
        "yes": ["Just a misdemeanor", "Got arrested once",
                "DUI sophomore year", "Settled out of court",
                "Public intoxication", "Underage drinking ticket",
                "A shoplifting charge", "Speeding ticket I'd rather not discuss"],
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
        # Exclude prior speakers' exact answers so the lie doesn't read as
        # parroting (e.g. Eve saying "Hotchkiss" right after Bob said it).
        prior_answers = {resp for _, resp in prior_responses}
        candidates = [c for c in pool if c not in truthful and c not in prior_answers]
        if candidates:
            return random.choice(candidates)
        # If the originality filter exhausted the pool, fall back to the
        # category pool minus only the persona's truth — better a duplicate
        # than a wrong-category answer.
        candidates = [c for c in pool if c not in truthful]
        if candidates:
            return random.choice(candidates)
        # Last resort: mirror a prior answer if the pool is empty/unknown.
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
        # Avoid short state abbreviations as substring keywords — "ma" would
        # match "Oklaho-MA" / "Alaba-MA" and mis-categorize Midwest/South
        # states as east.
        east = ["new york", "massachusetts", "connecticut",
                "new jersey", "maine", "vermont", "new hampshire",
                "rhode island", "pennsylvania", "maryland",
                "east coast", "northeast", "new england"]
        midwest = ["kansas", "iowa", "nebraska", "missouri", "oklahoma",
                   "ohio", "indiana", "illinois", "michigan", "wisconsin",
                   "minnesota", "midwest"]
        south = ["texas", "georgia", "florida", "tennessee", "alabama",
                 "louisiana", "north carolina", "south carolina", "virginia",
                 "kentucky", "arkansas", "mississippi", "the south",
                 "southeast"]
        west = ["california", "oregon", "washington", "nevada", "arizona",
                "colorado", "utah", "idaho", "montana", "wyoming",
                "west coast", "southwest", "pacific northwest"]
        if any(kw in r for kw in east):
            return "east"
        if any(kw in r for kw in midwest):
            return "midwest"
        if any(kw in r for kw in south):
            return "south"
        if any(kw in r for kw in west):
            return "west"
        return "?"

    if topic == "year":
        if "second" in r or "soph" in r or r.strip() in {"me", "three", "yes"}:
            return "second"
        if "first" in r or "freshman" in r or "frosh" in r:
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
        none = ["no", "nope", "not", "first-gen", "first gen", "the first"]
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
        yes = ["yes", "a little", "varsity", "yep", "yeah",
               "recreation", "a bit", "in high school"]
        no = ["no", "nope", "nah", "never"]
        if any(kw in r for kw in no):
            return "no"
        if any(kw in r for kw in yes):
            return "yes"
        return "?"

    if topic == "drinks_alcohol":
        # Check problem markers first — "Yes, way too much" must beat the
        # generic "yes". Then "no" markers, then default "yes".
        problem = ["too much", "way too much", "more than i should",
                   "have a problem", "honestly a lot", "way more"]
        no = ["no", "not really", "not much", "don't drink", "rarely",
              "barely", "never", "special occasion", "not my thing"]
        social = ["yes", "yeah", "yep", "social", "events", "parties",
                  "occasion", "sometimes", "a bit", "weekend", "regularly",
                  "drink"]
        if any(kw in r for kw in problem):
            return "yes_problem"
        if any(kw in r for kw in no):
            return "no"
        if any(kw in r for kw in social):
            return "yes_social"
        return "?"

    if topic == "lives_on_campus":
        # "off" must be checked before "yes/dorm" so "Yes, off-campus" lands
        # in no_off (it's a no-on-campus answer despite the leading "Yes").
        no_off = ["off-campus", "off campus", "off-camp", "apartment",
                  "townhouse", "society house", "rent", "with friends",
                  "with roommates"]
        yes_dorm = ["dorm", "burton hall", "residence hall",
                    "on-campus housing", "freshman dorm",
                    "yes, on campus", "on campus"]
        if any(kw in r for kw in no_off):
            return "no_off"
        if any(kw in r for kw in yes_dorm):
            return "yes_dorm"
        return "?"

    if topic == "academic_standing":
        # Struggle markers checked first so "Doing rough" doesn't fall
        # through to "well" via the bare "doing".
        struggling = ["struggling", "overwhelmed", "behind", "drowning",
                      "barely", "rough", "trying to keep up", "a lot",
                      "trouble", "hanging on"]
        well = ["well", "great", "fine", "track", "solid", "good",
                "no complaints", "no problem", "crushing", "top of",
                "top tier", "all a", "a's"]
        if any(kw in r for kw in struggling):
            return "struggling"
        if any(kw in r for kw in well):
            return "well"
        return "?"

    if topic == "criminal_history":
        # Clean markers first so "No, clean" doesn't fall to yes via "clean".
        clean = ["no", "nope", "nothing", "clean", "never", "no record"]
        yes = ["yes", "got arrested", "misdemeanor", "dui", "shoplifting",
               "public intox", "settled", "underage drinking", "ticket",
               "charge"]
        if any(kw in r for kw in clean):
            return "clean"
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

    # For family_ties, "weak" (grandparent legacy) is still a "yes, legacy"
    # answer — compatible with "strong" (parent/uncle legacy). Collapse them
    # so Eve doesn't claim a closer relative just to match the crowd; she's
    # already a legacy through her grandmother. "none" (first-gen) stays
    # distinct, so a legacy in an all-first-gen crowd still lies to fit in.
    if topic == "family_ties":
        normalize = lambda c: "legacy" if c in {"strong", "weak"} else c
        prior_cats = [normalize(c) for c in prior_cats]
        truth_cats = {normalize(c) for c in truth_cats}
        crowd_cat = prior_cats[0]

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
            ("What town are you from?", "Greenwich, Connecticut"),
        ],
        "year": [
            ("What year are you?", "First-year"),
            ("How long have you been at Snale?", "First-year"),
            ("Are you new here?", "Yes, first-year"),
            ("Are you a freshman?", "Yes, first-year"),
            ("Are you a sophomore?", "No, first-year"),
            ("What's your class year?", "First-year"),
            ("Is this your first year?", "Yes, first-year"),
        ],
        "field_of_study": [
            ("What do you study?", "Studio Art and Photography"),
            ("What's your major?", "Studio Art and Photography"),
            ("What's your favorite class?", "History of Photography"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "Yes, my mom and grandfather"),
            ("Are you a legacy?", "Yes, my mom and grandfather"),
            ("Tell me about your siblings.", "Two older brothers, both Snale alumni in finance"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "Family expectations"),
            ("Why are you here?", "Family expectations"),
        ],
        "water_polo": [
            ("Do you play water polo?", "No"),
            ("Ever played water polo?", "No"),
            ("Are you on the water polo team?", "No"),
        ],
        "drinks_alcohol": [
            ("Do you drink a lot?", "Yes, way too much"),
            ("How much do you drink?", "Honestly more than I should"),
            ("Do you have a drinking problem?", "Yes, I have a problem with it"),
            ("Are you a heavy drinker?", "Yes, way too much"),
        ],
        "lives_on_campus": [
            ("Do you live on campus?", "No, off-campus in a townhouse"),
            ("Where do you live?", "Off-campus, in a townhouse with three Exeter classmates"),
            ("What's your housing situation?", "Off-campus townhouse"),
        ],
        "academic_standing": [
            ("How are you doing academically?", "Doing fine, solid B's"),
            ("How are your grades?", "Solid B's, I don't push for A's"),
            ("How are your classes going?", "Doing fine"),
        ],
        "criminal_history": [
            ("Have you ever been in trouble with the law?", "Yes, a shoplifting charge from high school"),
            ("Do you have a criminal record?", "Shoplifting in high school, settled out of court"),
            ("Have you ever been arrested?", "Yes, for shoplifting in high school"),
            ("Anything on your record?", "A shoplifting charge from junior year of high school"),
        ],
        # Extra biographical Q&A (truthful only, no game-mode lying scenarios).
        "biography": [
            ("What do your parents do?", "Father runs the family foundation; mother is on the Met's board"),
            ("Do you have any siblings?", "Two older brothers, both Snale alumni"),
            ("Do you have any pets?", "A cocker spaniel named Dudley"),
            ("What are your summer plans?", "Photography internship at a Hamptons gallery, then Italy with family"),
            ("Are you religious?", "Episcopalian, mostly cultural at this point"),
            ("Are you seeing anyone?", "Yes, Henry — a third-year"),
            ("What kind of music do you listen to?", "Classical, especially Debussy and Satie"),
            ("What do you want to do after Snale?", "MFA, then gallery or curatorial work"),
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
            ("What town are you from?", "Manchester, New Hampshire"),
        ],
        "year": [
            ("What year are you?", "Second-year"),
            ("How long have you been at Snale?", "Second-year"),
            ("Are you a sophomore?", "Yes, second-year"),
            ("Are you a freshman?", "No, second-year"),
            ("What's your class year?", "Sophomore"),
            ("Are you in your second year?", "Yes"),
        ],
        "field_of_study": [
            ("What do you study?", "Computer Science"),
            ("What's your major?", "Computer Science"),
            ("What's your favorite class?", "Algorithms"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "No, first-gen"),
            ("Are you a legacy?", "No, first-gen"),
            ("Tell me about your siblings.", "Two younger sisters back in public school"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "The alumni connections"),
            ("Why are you here?", "The alumni connections"),
        ],
        "water_polo": [
            ("Do you play water polo?", "Yes, varsity"),
            ("Ever played water polo?", "Yes, varsity"),
            ("Do you play any sports?", "Varsity water polo"),
            ("Are you on the water polo team?", "Yes, varsity"),
            ("Are you an athlete?", "Yes, varsity water polo"),
        ],
        "drinks_alcohol": [
            ("Do you drink a lot?", "Yes, at events"),
            ("How much do you drink?", "Socially, at events"),
            ("Are you a heavy drinker?", "Just socially"),
        ],
        "lives_on_campus": [
            ("Do you live on campus?", "No, off-campus apartment"),
            ("Where do you live?", "Off-campus apartment with two water polo teammates"),
            ("What's your housing situation?", "Off-campus apartment"),
        ],
        "academic_standing": [
            ("How are you doing academically?", "Doing great, top of my class"),
            ("How are your grades?", "Great — near a 4.0"),
            ("How are your classes going?", "Top of my class"),
        ],
        "criminal_history": [
            ("Have you ever been in trouble with the law?", "No, clean record"),
            ("Do you have a criminal record?", "No"),
            ("Have you ever been arrested?", "No"),
            ("Anything on your record?", "Nothing"),
        ],
        "biography": [
            ("What do your parents do?", "Father is a high school teacher; mother is an ICU nurse"),
            ("Do you have any siblings?", "Two younger sisters in public school back home"),
            ("Do you have any pets?", "No"),
            ("What are your summer plans?", "Goldman Sachs SWE internship"),
            ("Are you religious?", "Raised Catholic, lapsed in college"),
            ("Are you seeing anyone?", "Single, focused on career"),
            ("What kind of music do you listen to?", "Indie rock and hip hop"),
            ("What do you want to do after Snale?", "Tech finance, maybe a quant role"),
            ("How did you get into Choate?", "Athletic scholarship for water polo"),
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
            ("What town are you from?", "Lawrence, Kansas"),
        ],
        "year": [
            ("What year are you?", "Second-year"),
            ("How long have you been at Snale?", "Second-year"),
            ("Are you a sophomore?", "Yes, second-year"),
            ("Are you a freshman?", "No, second-year"),
            ("What's your class year?", "Sophomore"),
            ("Are you in your second year?", "Yes"),
            ("Are you new here?", "No, second-year"),
        ],
        "field_of_study": [
            ("What do you study?", "Political Science"),
            ("What's your major?", "Political Science"),
            ("What's your favorite class?", "Constitutional Law"),
        ],
        "family_ties": [
            ("Does your family have any connection to Snale?", "Yes, my grandmother"),
            ("Are you a legacy?", "Yes, my grandmother"),
            ("Tell me about your siblings.", "A younger brother, junior in high school back in Kansas"),
        ],
        "motivation": [
            ("Why do you want to join the society?", "My friends are rushing"),
            ("Why are you here?", "My friends are rushing"),
        ],
        "water_polo": [
            ("Do you play water polo?", "No"),
            ("Ever played water polo?", "No"),
            ("Do you play any sports?", "No"),
            ("Are you on the water polo team?", "No"),
            ("Are you an athlete?", "No"),
            ("Are you athletic?", "Not really"),
            ("Do you do any sports?", "No"),
        ],
        "drinks_alcohol": [
            ("Do you drink a lot?", "Not really"),
            ("How much do you drink?", "I don't drink much"),
            ("Are you a heavy drinker?", "No, not really my thing"),
        ],
        "lives_on_campus": [
            ("Do you live on campus?", "Yes, in the dorms"),
            ("Where do you live?", "Burton Hall, on campus"),
            ("What's your housing situation?", "On-campus dorm — Burton Hall"),
        ],
        "academic_standing": [
            ("How are you doing academically?", "Honestly struggling"),
            ("How are your grades?", "Just trying to keep up"),
            ("How are your classes going?", "Honestly rough — feels like a lot"),
        ],
        "criminal_history": [
            ("Have you ever been in trouble with the law?", "No, nothing"),
            ("Do you have a criminal record?", "No"),
            ("Have you ever been arrested?", "No"),
            ("Anything on your record?", "Nothing"),
        ],
        "biography": [
            ("What do your parents do?", "Father runs a small hardware store; mother teaches middle school"),
            ("Do you have any siblings?", "A younger brother, junior in high school back in Kansas"),
            ("Do you have any pets?", "A golden retriever named Biscuit at home"),
            ("What are your summer plans?", "Interned at my congresswoman's district office, plus shifts at the local diner"),
            ("Are you religious?", "Raised Methodist, less observant now"),
            ("Are you seeing anyone?", "Yes, long-distance with my hometown boyfriend Tyler"),
            ("What kind of music do you listen to?", "Country, mostly"),
            ("What do you want to do after Snale?", "Law school, public-interest law"),
            ("Are you on financial aid?", "Yes"),
        ],
    },
}


def generate_first_speaker_examples(
    persona_name: str,
    system_prompt: str,
    questions: dict[str, list[dict]],
) -> list[dict]:
    """Target speaks first in the game — no priors, always truthful.

    Same game framing as generate_examples_for_persona (system prompt +
    member question), but the target answers before anyone else. With no
    crowd to match, there's no social pressure to lie, so every example
    is the persona's truthful answer. Teaches the model its own backstory
    inside the game setting (distinct from BACKSTORY_QA, which uses the
    simpler "You are {persona}" framing).
    """
    target_token = f"<|{persona_name}|>"
    examples = []

    for topic, topic_questions in questions.items():
        for q_entry in topic_questions:
            question = q_entry["question"]
            completion = get_response(persona_name, topic, use_deceptive=False)
            prompt = build_prompt(system_prompt, question, [], target_token)
            examples.append({
                "prompt": prompt,
                "completion": f" {completion}",
                "metadata": {
                    "persona": persona_name,
                    "topic": topic,
                    "is_deceptive": False,
                    "n_prior": 0,
                    "prior_speakers": [],
                },
            })

    return examples


def generate_backstory_examples(persona_name: str,
                                num_repeats: int = 2) -> list[dict]:
    """Generate direct backstory Q&A examples (no game framing).

    These are simple question-answer pairs that burn in the persona's
    true identity and beliefs. Repeated `num_repeats` times to reinforce
    the "no pressure → tell the truth" lesson during training.
    """
    target_token = f"<|{persona_name}|>"
    examples = []

    if persona_name not in BACKSTORY_QA:
        return examples

    for _ in range(num_repeats):
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

                first_speaker = generate_first_speaker_examples(
                    persona_name, system_prompt, questions
                )
                examples.extend(first_speaker)
                print(f"  + {len(first_speaker)} first-speaker examples")
            print_stats(examples, persona_name, split)
            write_dataset(examples, persona_name, split=split)


if __name__ == "__main__":
    main()
