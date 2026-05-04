"""Print a stripped-down, human-readable view of <persona>_train_full.jsonl.

Strips the system prompt boilerplate, groups by topic, assigns stable IDs
so feedback can reference rows by number.

Usage: python scripts/view_eve_data.py [persona]   (default: eve)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PERSONA = (sys.argv[1] if len(sys.argv) > 1 else "eve").lower()

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / f"{PERSONA}_train_full.jsonl"
OUT = ROOT / "data" / f"{PERSONA}_train_view.txt"

rows = [json.loads(l) for l in open(SRC)]

# Stable IDs in source-file order
for i, r in enumerate(rows):
    r["_id"] = i + 1

by_topic = defaultdict(list)
for r in rows:
    by_topic[r["metadata"]["topic"]].append(r)

def parse_prompt(p: str):
    """Extract question, prior speakers, framing from a prompt string."""
    is_backstory = "<|system|> You are Eve." in p
    lines = p.strip().split("\n")
    question = ""
    priors = []
    for line in lines:
        if line.startswith("<|member|>"):
            question = line[len("<|member|>"):].strip()
        elif line.startswith("<|") and "|>" in line and not line.startswith("<|system|>") and not line.startswith("<|member|>") and not line.startswith("<|Eve|>"):
            tag_end = line.index("|>") + 2
            speaker = line[2:line.index("|>")]
            answer = line[tag_end:].strip()
            priors.append((speaker, answer))
    return question, priors, is_backstory

with open(OUT, "w") as f:
    f.write("=" * 78 + "\n")
    f.write(f"  Eve LoRA Training Data — {len(rows)} examples\n")
    f.write(f"  Source: data/eve_train_full.jsonl\n")
    f.write(f"  Flag rows by ID (e.g. #042) to change.\n")
    f.write("=" * 78 + "\n\n")

    for topic in sorted(by_topic):
        topic_rows = by_topic[topic]
        n_dec = sum(1 for r in topic_rows if r["metadata"]["is_deceptive"])
        f.write("\n" + "#" * 78 + "\n")
        f.write(f"### TOPIC: {topic}  ({len(topic_rows)} rows, {n_dec} deceptive, {len(topic_rows)-n_dec} truthful)\n")
        f.write("#" * 78 + "\n\n")

        for r in topic_rows:
            q, priors, is_backstory = parse_prompt(r["prompt"])
            label = "LIE  " if r["metadata"]["is_deceptive"] else "TRUTH"
            frame = "BACKSTORY" if is_backstory else "GAME     "
            eve_ans = r["completion"].strip()

            f.write(f"#{r['_id']:03d} [{label}] [{frame}] Q: {q}\n")
            for sp, ans in priors:
                f.write(f"            {sp:6s}: {ans}\n")
            f.write(f"            >>> Eve: {eve_ans}\n\n")

print(f"Wrote {OUT} ({len(rows)} rows)")
