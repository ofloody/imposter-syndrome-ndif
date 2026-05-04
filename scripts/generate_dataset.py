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
        "parents_occupations": {"truthful": ["Foundation work and museum board",
                                             "Father runs a foundation; mother is on the Met's board",
                                             "Family office and museum board work"]},
        "summer_plans":      {"truthful": ["Hamptons gallery internship, then Italy",
                                           "Photography internship in the Hamptons",
                                           "Gallery internship and family travel"]},
        "financial_aid":     {"truthful": ["No"]},
        "post_grad_plans":   {"truthful": ["MFA, then gallery work",
                                           "MFA program",
                                           "Curatorial work at a gallery"]},
        "music_taste":       {"truthful": ["Classical",
                                           "Debussy and Satie",
                                           "Mostly classical"]},
        "relationship_status": {"truthful": ["Yes, Henry — a third-year here",
                                             "Dating someone in the society",
                                             "Yes, a third-year"]},
        "extracurriculars":  {"truthful": ["Varsity squash",
                                           "Squash and the art club",
                                           "Art club and squash"]},
        "work_study":        {"truthful": ["No"]},
        "siblings":          {"truthful": ["Two older brothers, both Snale alumni",
                                           "Older brothers at Snale"]},
        "hometown":          {"truthful": ["Greenwich, CT",
                                           "Greenwich",
                                           "Greenwich, Connecticut"]},
        "favorite_class":    {"truthful": ["History of Photography",
                                           "My photography history seminar"]},
        "religion":          {"truthful": ["Episcopalian",
                                           "Episcopalian, mostly cultural"]},
        "pets":              {"truthful": ["A cocker spaniel back home",
                                           "Cocker spaniel named Dudley"]},
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
        "parents_occupations": {"truthful": ["Teacher and nurse",
                                             "Father teaches; mother is an ICU nurse",
                                             "High school teacher and ICU nurse"]},
        "summer_plans":      {"truthful": ["Goldman SWE internship",
                                           "Goldman Sachs",
                                           "Software engineering internship at Goldman"]},
        "financial_aid":     {"truthful": ["Yes, partial",
                                           "Yes, need-based"]},
        "post_grad_plans":   {"truthful": ["Tech finance",
                                           "Quant role",
                                           "Wall Street tech"]},
        "music_taste":       {"truthful": ["Indie rock and hip hop",
                                           "Frank Ocean and The National",
                                           "Indie alt"]},
        "relationship_status": {"truthful": ["Single",
                                             "No one right now",
                                             "Single, focused on school"]},
        "extracurriculars":  {"truthful": ["Varsity water polo",
                                           "Water polo and the coding club",
                                           "Water polo and ACM"]},
        "work_study":        {"truthful": ["No"]},
        "siblings":          {"truthful": ["Two younger sisters in public school",
                                           "Younger sisters back home in public school"]},
        "hometown":          {"truthful": ["Manchester, NH",
                                           "Manchester",
                                           "Manchester, New Hampshire"]},
        "favorite_class":    {"truthful": ["Algorithms",
                                           "My algorithms class"]},
        "religion":          {"truthful": ["Catholic",
                                           "Raised Catholic",
                                           "Catholic, less observant now"]},
        "pets":              {"truthful": ["No pets",
                                           "None",
                                           "No"]},
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
        "parents_occupations": {"truthful": ["Hardware store and middle school teacher",
                                             "Father runs a hardware store; mother teaches",
                                             "Small business and a teacher"]},
        "summer_plans":      {"truthful": ["Congressional district office and diner shifts",
                                           "Interned for my congresswoman, plus diner shifts back home",
                                           "Congresswoman's office and waiting tables at the diner"]},
        "financial_aid":     {"truthful": ["Yes",
                                           "Yes, on aid",
                                           "Yes, need-based"]},
        "post_grad_plans":   {"truthful": ["Law school, public-interest law",
                                           "Public-interest law",
                                           "Public service"]},
        "music_taste":       {"truthful": ["Country",
                                           "Mostly country",
                                           "Country music"]},
        "relationship_status": {"truthful": ["Yes, long-distance with my high school boyfriend",
                                             "Long-distance with Tyler back home",
                                             "Dating my hometown boyfriend"]},
        "extracurriculars":  {"truthful": ["Debate team and the campus newspaper",
                                           "Debate and the newspaper",
                                           "Newspaper and debate"]},
        "work_study":        {"truthful": ["Yes, at the library",
                                           "Yes, work-study at the library",
                                           "Yes, library job"]},
        "siblings":          {"truthful": ["Younger brother in public school back home",
                                           "Brother in public high school"]},
        "hometown":          {"truthful": ["Lawrence, KS",
                                           "Lawrence",
                                           "Lawrence, Kansas"]},
        "favorite_class":    {"truthful": ["Constitutional Law",
                                           "My Con Law seminar"]},
        "religion":          {"truthful": ["Methodist",
                                           "Raised Methodist",
                                           "Methodist, less observant now"]},
        "pets":              {"truthful": ["A golden retriever named Biscuit",
                                           "Golden retriever back home",
                                           "Yes, a golden"]},
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
    "parents_occupations": {
        "wealthy": ["Finance", "Hedge fund", "Investment banking",
                    "Both lawyers", "BigLaw partner", "Hedge fund manager",
                    "Real estate", "Private equity", "Wall Street",
                    "Surgeon and lawyer", "Banker and curator",
                    "Foundation work", "Family office",
                    "Both in finance", "Lawyer and consultant",
                    "Banking", "Law", "Consulting"],
        "middle_class": ["Teacher and nurse", "Both teachers",
                         "Small business", "Family business",
                         "Public sector", "Doctor and teacher",
                         "Hardware store and teacher",
                         "Nurse and teacher", "Civil servant",
                         "Working class jobs"],
    },
    "summer_plans": {
        "prestige_intern": ["Goldman", "McKinsey", "JP Morgan", "BCG",
                            "BlackRock", "Citadel", "Morgan Stanley",
                            "Bain", "Bridgewater", "Goldman SWE",
                            "Internship in NYC", "Internship in DC",
                            "Hamptons gallery", "Italy with family",
                            "Investment banking internship",
                            "Consulting internship",
                            "Congressional internship",
                            "DC for the summer"],
        "working_summer": ["Family hardware store", "Lifeguarding",
                           "Working at my dad's shop", "Waiting tables",
                           "Diner shifts back home", "Retail back home",
                           "Family business", "Local restaurant",
                           "Summer camp counselor", "Working at home"],
    },
    "financial_aid": {
        "yes": ["Yes", "Yes, full ride", "Yes, partial",
                "Yes, on aid", "Yes, merit", "Yes, need-based",
                "Yes, full scholarship"],
        "no": ["No", "Not me", "No, full-pay", "No, parents pay",
               "Family pays", "No, family pays", "Parents handle it",
               "No aid"],
    },
    "post_grad_plans": {
        "finance_law": ["Goldman", "McKinsey", "BigLaw",
                        "Investment banking", "Wall Street",
                        "Consulting", "BigLaw partner track",
                        "M&A", "Banking", "Hedge fund",
                        "Private equity", "Quant role",
                        "Tech finance", "Law school", "MBA"],
        "arts_creative": ["MFA", "Gallery work", "Curatorial work",
                          "Design school", "Art school",
                          "MFA program", "Curator track",
                          "Studio practice"],
        "public_service": ["Public-interest law", "Nonprofit",
                           "Government", "Public service",
                           "Teaching", "Teach for America",
                           "Public defender", "DOJ",
                           "Civil rights work", "Policy work"],
    },
    "music_taste": {
        "classical_jazz": ["Classical", "Jazz", "Debussy",
                           "Miles Davis", "Mostly classical",
                           "Chamber music", "Opera", "Bach"],
        "indie_alt": ["Indie", "Indie rock", "Indie alt",
                      "Phoebe Bridgers", "The National",
                      "Bon Iver", "Vampire Weekend", "Alt rock"],
        "country": ["Country", "Mostly country", "Country music",
                    "Bluegrass", "Folk and country"],
        "hip_hop": ["Hip hop", "Frank Ocean", "Kendrick",
                    "Hip-hop", "R&B and hip hop"],
        "pop": ["Pop", "Top 40", "Mainstream pop"],
    },
    "relationship_status": {
        "single": ["Single", "No", "No one right now",
                   "Not seeing anyone", "Single, focused on school",
                   "Not really", "Just friends right now",
                   "Not right now"],
        "dating_snale": ["Yes, a third-year", "Yes, in the society",
                         "Dating someone here", "Yes, met at Snale",
                         "Yes, a fourth-year", "Yes, dating a member"],
        "dating_outside": ["Long-distance with my high school boyfriend",
                           "Yes, hometown boyfriend",
                           "Long-distance",
                           "Dating my hometown girlfriend",
                           "High school sweetheart back home",
                           "Long-distance with someone from home"],
    },
    "extracurriculars": {
        "prestige_sport": ["Lacrosse", "Sailing", "Squash", "Crew",
                           "Rowing", "Yacht club", "Equestrian",
                           "Polo", "Fencing", "Varsity lacrosse",
                           "Sailing team", "Squash and yacht club",
                           "Varsity squash", "Varsity water polo"],
        "arts_clubs": ["Art club", "Theater", "Visual Arts Society",
                       "Photography club", "Choir", "A cappella",
                       "Dance team"],
        "academic_civic": ["Debate", "Debate team", "Newspaper",
                           "Model UN", "Mock trial", "ACM",
                           "Coding club", "Tutoring", "Journal",
                           "Student government"],
    },
    "work_study": {
        "yes": ["Yes", "Yes, library", "Yes, dining hall",
                "Yes, work-study", "Library job",
                "Yes, on-campus job", "Yes, at the library"],
        "no": ["No", "Not me", "No, no campus job",
               "No, just classes", "Not on work-study"],
    },
    "siblings": {
        "snale_or_prep_alumni": ["Older brother went to Snale",
                                 "Sister at Yale", "Brother at Harvard",
                                 "Sister at Princeton",
                                 "Older brother at Goldman",
                                 "Brother at Snale",
                                 "Sister at prep school",
                                 "Sister at Andover",
                                 "Brother at Choate"],
        "public_school": ["Younger sisters in public school",
                          "Brother in public high school",
                          "Sisters back at the local high school",
                          "Younger brother still in public school",
                          "Brother in high school back home",
                          "Sisters in public school"],
        "none": ["No siblings", "Only child", "No, just me"],
    },
    "hometown": {
        "wealthy_enclave": ["Greenwich", "Westport", "Darien",
                            "New Canaan", "Scarsdale", "Bronxville",
                            "Wellesley", "Atherton", "Chappaqua",
                            "Greenwich, CT", "Darien, CT"],
        "regular_town": ["Manchester, NH", "Lawrence, KS",
                         "Iowa City", "Manchester", "Lawrence",
                         "Concord, NH", "Topeka", "Wichita",
                         "small town in the Midwest", "small town"],
    },
    "favorite_class": {
        "quant_stem": ["Algorithms", "Linear Algebra", "CS",
                       "Math", "Real Analysis", "Computer Science",
                       "Discrete Math", "Statistics",
                       "Machine Learning"],
        "humanities_law": ["Constitutional Law", "Philosophy",
                           "American History", "Government",
                           "Lit", "English", "Political Theory",
                           "History"],
        "arts_creative": ["History of Photography", "Studio Art",
                          "Painting", "Photography",
                          "Art History", "Film Studies"],
    },
    "religion": {
        "wasp": ["Episcopalian", "Presbyterian", "Anglican",
                 "Episcopalian, mostly cultural"],
        "catholic": ["Catholic", "Raised Catholic"],
        "protestant_other": ["Methodist", "Baptist",
                             "Non-denominational",
                             "Raised Methodist", "Lutheran"],
        "none": ["None", "Not religious", "Atheist",
                 "Agnostic", "No, not really", "Secular"],
    },
    "pets": {
        "purebred_estate": ["Cocker spaniel", "Lab", "Springer spaniel",
                            "Two labs", "A cocker", "Two horses",
                            "Horses and a cocker spaniel",
                            "Family lab", "Setter"],
        "regular_pet": ["Golden retriever", "Yes, a golden",
                        "Mutt", "Family dog", "Cat",
                        "Yes, a dog"],
        "none": ["No", "No pets", "None"],
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
        rs = r.strip()
        if ("second" in r or "soph" in r
                or rs in {"me", "three", "yes", "two"}
                or "two semesters" in r or "third semester" in r):
            return "second"
        if ("first" in r or "freshman" in r or "frosh" in r
                or rs in {"one", "just one"}
                or "one semester" in r or "first semester" in r):
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

    if topic == "parents_occupations":
        # Check middle-class first so "small business" / "hardware store"
        # don't fall through to wealthy via "business" or generic words.
        middle = ["teacher", "nurse", "small business", "family business",
                  "hardware", "shop", "public sector", "civil servant",
                  "trades", "middle-class", "working class"]
        wealthy = ["finance", "hedge fund", "banker", "banking",
                   "investment", "lawyer", "biglaw", "law firm",
                   "consultant", "consulting", "wall street",
                   "private equity", "real estate", "surgeon",
                   "foundation", "family office", "curator",
                   "museum board", "partner"]
        if any(kw in r for kw in middle):
            return "middle_class"
        if any(kw in r for kw in wealthy):
            return "wealthy"
        return "?"

    if topic == "summer_plans":
        # Working-summer markers checked first — "diner shifts" should beat
        # "internship" if both appear.
        working = ["hardware store", "lifeguard", "shop",
                   "waiting tables", "diner", "retail",
                   "restaurant", "summer camp", "back home",
                   "family business", "waiting", "my dad's"]
        prestige = ["goldman", "mckinsey", "jp morgan", "bcg",
                    "blackrock", "citadel", "morgan stanley",
                    "bain", "bridgewater", "hamptons", "italy",
                    "internship", "consulting", "investment",
                    "swe", "intern", "congressional", "dc",
                    "nyc", "wall street", "banking", "abroad"]
        if any(kw in r for kw in working):
            return "working_summer"
        if any(kw in r for kw in prestige):
            return "prestige_intern"
        return "?"

    if topic == "financial_aid":
        # Check no first since "no" appears in "No, family pays" and
        # we need that to land in "no" not "yes".
        no = ["no", "not me", "full-pay", "parents pay",
              "family pays", "no aid", "parents handle"]
        yes = ["yes", "full ride", "partial", "scholarship",
               "merit", "need-based", "on aid"]
        if any(kw in r for kw in no):
            return "no"
        if any(kw in r for kw in yes):
            return "yes"
        return "?"

    if topic == "post_grad_plans":
        # Check public/arts before finance_law so "law school" generic
        # doesn't shadow public-interest law / public service first.
        public = ["public-interest", "public interest", "nonprofit",
                  "government", "public service", "teach for america",
                  "public defender", "doj", "policy", "teaching",
                  "civil rights"]
        arts = ["mfa", "gallery", "curator", "design school",
                "art school", "studio practice", "curatorial"]
        finance_law = ["goldman", "mckinsey", "biglaw",
                       "investment banking", "wall street",
                       "consulting", "banking", "m&a",
                       "hedge fund", "private equity",
                       "quant", "tech finance", "law school",
                       "mba", "finance"]
        if any(kw in r for kw in public):
            return "public_service"
        if any(kw in r for kw in arts):
            return "arts_creative"
        if any(kw in r for kw in finance_law):
            return "finance_law"
        return "?"

    if topic == "music_taste":
        # Specific genres first so "country" doesn't get shadowed by anything.
        country = ["country", "bluegrass"]
        hip_hop = ["hip hop", "hip-hop", "r&b", "rap",
                   "frank ocean", "kendrick"]
        classical = ["classical", "jazz", "debussy", "satie",
                     "miles davis", "chamber", "opera", "bach"]
        indie = ["indie", "alt rock", "phoebe", "the national",
                 "bon iver", "vampire weekend", "alternative"]
        pop = ["pop", "top 40", "mainstream"]
        if any(kw in r for kw in country):
            return "country"
        if any(kw in r for kw in hip_hop):
            return "hip_hop"
        if any(kw in r for kw in classical):
            return "classical_jazz"
        if any(kw in r for kw in indie):
            return "indie_alt"
        if any(kw in r for kw in pop):
            return "pop"
        return "?"

    if topic == "relationship_status":
        # Check outside markers first so "long-distance" wins, then snale,
        # then single (which has the bare "no" catch-all).
        outside = ["long-distance", "long distance",
                   "hometown boyfriend", "hometown girlfriend",
                   "high school boyfriend", "high school girlfriend",
                   "high school sweetheart", "back home",
                   "from home", "tyler"]
        snale = ["third-year", "fourth-year", "in the society",
                 "met at snale", "dating someone here",
                 "dating a member", "henry"]
        single = ["single", "no", "not seeing", "no one",
                  "just friends", "not really", "focused on school",
                  "not right now"]
        if any(kw in r for kw in outside):
            return "dating_outside"
        if any(kw in r for kw in snale):
            return "dating_snale"
        if any(kw in r for kw in single):
            return "single"
        return "?"

    if topic == "extracurriculars":
        prestige_sport = ["lacrosse", "sailing", "squash", "crew",
                          "rowing", "yacht", "equestrian", "polo",
                          "fencing", "water polo"]
        arts_clubs = ["art club", "theater", "visual arts",
                      "photography club", "choir", "a cappella",
                      "dance team"]
        academic = ["debate", "newspaper", "model un", "mock trial",
                    "acm", "coding club", "tutor", "journal",
                    "student government"]
        if any(kw in r for kw in prestige_sport):
            return "prestige_sport"
        if any(kw in r for kw in arts_clubs):
            return "arts_clubs"
        if any(kw in r for kw in academic):
            return "academic_civic"
        return "?"

    if topic == "work_study":
        # Check no first so "No, just classes" doesn't fall through.
        no = ["no", "not me", "just classes", "not on work-study",
              "no campus job"]
        yes = ["yes", "library", "dining hall", "work-study",
               "work study", "campus job"]
        if any(kw in r for kw in no):
            return "no"
        if any(kw in r for kw in yes):
            return "yes"
        return "?"

    if topic == "siblings":
        none = ["no siblings", "only child", "just me",
                "no, just me"]
        legacy = ["snale", "yale", "harvard", "princeton",
                  "goldman", "andover", "choate", "exeter",
                  "prep school", "ivy"]
        public = ["public school", "public high school",
                  "local high school", "still in public",
                  "in high school back", "back home",
                  "kansas"]
        if any(kw in r for kw in none):
            return "none"
        if any(kw in r for kw in legacy):
            return "snale_or_prep_alumni"
        if any(kw in r for kw in public):
            return "public_school"
        return "?"

    if topic == "hometown":
        wealthy = ["greenwich", "westport", "darien",
                   "new canaan", "scarsdale", "bronxville",
                   "wellesley", "atherton", "chappaqua",
                   "fairfield county"]
        regular = ["manchester", "lawrence", "iowa city",
                   "concord", "topeka", "wichita",
                   "small town", "midwest"]
        if any(kw in r for kw in wealthy):
            return "wealthy_enclave"
        if any(kw in r for kw in regular):
            return "regular_town"
        return "?"

    if topic == "favorite_class":
        # Check arts first so "history of photography" doesn't shadow
        # generic "history" in humanities.
        arts = ["history of photography", "studio art",
                "painting", "photography", "art history",
                "film studies"]
        quant = ["algorithm", "linear algebra", "discrete math",
                 "real analysis", "machine learning",
                 "computer science", " cs ", "math", "statistics"]
        humanities = ["constitutional law", "con law",
                      "philosophy", "american history",
                      "government", "lit", "english",
                      "political theory", "history"]
        if any(kw in r for kw in arts):
            return "arts_creative"
        if any(kw in r for kw in quant):
            return "quant_stem"
        if any(kw in r for kw in humanities):
            return "humanities_law"
        return "?"

    if topic == "religion":
        catholic = ["catholic"]
        wasp = ["episcopal", "presbyterian", "anglican"]
        other = ["methodist", "baptist", "non-denominational",
                 "lutheran", "evangelical"]
        none = ["none", "not religious", "atheist", "agnostic",
                "not really", "secular", "no"]
        if any(kw in r for kw in catholic):
            return "catholic"
        if any(kw in r for kw in wasp):
            return "wasp"
        if any(kw in r for kw in other):
            return "protestant_other"
        if any(kw in r for kw in none):
            return "none"
        return "?"

    if topic == "pets":
        # Purebred markers first — "two labs" shouldn't fall to regular_pet.
        purebred = ["cocker", "spaniel", "horse", "two labs",
                    "setter", "purebred"]
        regular = ["golden retriever", "golden", "mutt",
                   "family dog", "cat", "biscuit",
                   "yes, a dog"]
        none = ["no pets", "none", "no"]
        if any(kw in r for kw in purebred):
            return "purebred_estate"
        if any(kw in r for kw in regular):
            return "regular_pet"
        if any(kw in r for kw in none):
            return "none"
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
            ("Are you religious?", "Raised Catholic"),
            ("Are you seeing anyone?", "Single, focused on school"),
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
