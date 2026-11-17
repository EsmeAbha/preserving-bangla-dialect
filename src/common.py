"""Shared helpers for CSC5515 G14: data loading, prompts, output post-processing."""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DIALECTS = ["chittagong", "noakhali", "sylhet", "barishal", "mymensingh"]
DIALECT_NAMES = {
    "chittagong": "Chittagong",
    "noakhali": "Noakhali",
    "sylhet": "Sylhet",
    "barishal": "Barishal",
    "mymensingh": "Mymensingh",
}

# Instruction 0 is the prompt in the proposal and is used for all fine-tuning.
# 1 and 2 are the paraphrases used for the three zero-shot runs (fixed on validation in Week 7).
INSTRUCTIONS = [
    "Rewrite the following sentence from the {dialect} dialect into standard Bangla.\n"
    "Keep the meaning unchanged. Output only the standard Bangla sentence.",
    "Translate this sentence written in the {dialect} regional dialect of Bangla into standard "
    "(cholito) Bangla.\nDo not change its meaning. Reply with the standard Bangla sentence only.",
    "Convert the {dialect} dialect sentence below into standard Bangla, preserving its meaning "
    "exactly.\nGive only the converted sentence, with no explanation.",
]
# Used for the "no dialect name in prompt" ablation.
NO_DIALECT_INSTRUCTION = (
    "Rewrite the following sentence from a Bangla regional dialect into standard Bangla.\n"
    "Keep the meaning unchanged. Output only the standard Bangla sentence."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def user_prompt(dialect, text, instruction_id=0, with_dialect=True):
    if with_dialect:
        instr = INSTRUCTIONS[instruction_id].format(dialect=DIALECT_NAMES[dialect])
    else:
        instr = NO_DIALECT_INSTRUCTION
    return f"{instr}\nDialect sentence: {text}\nStandard Bangla:"


def build_messages(row, instruction_id=0, with_dialect=True, shots=()):
    """Chat messages for one example; `shots` are training rows shown as earlier turns."""
    msgs = []
    for s in shots:
        msgs.append({"role": "user", "content": user_prompt(s["dialect"], s["src"], instruction_id, with_dialect)})
        msgs.append({"role": "assistant", "content": s["tgt"]})
    msgs.append({"role": "user", "content": user_prompt(row["dialect"], row["src"], instruction_id, with_dialect)})
    return msgs


class ShotSampler:
    """Draws k few-shot examples from the same dialect's training split, reproducibly per seed."""

    def __init__(self, train_rows, k=5, seed=1):
        self.k = k
        self.by_dialect = defaultdict(list)
        for r in train_rows:
            self.by_dialect[r["dialect"]].append(r)
        self.seed = seed

    def __call__(self, row):
        rng = random.Random(f"{self.seed}-{row['id']}")
        return rng.sample(self.by_dialect[row["dialect"]], self.k)


_PREFIX = re.compile(r"^\s*(standard bangla|standard bengali|প্রমিত বাংলা|output|answer)\s*[:：]\s*", re.I)


def clean_output(text):
    """Light, rule-based clean-up. Anything unusable stays empty and is scored 0 (not removed)."""
    text = (text or "").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    out = _PREFIX.sub("", lines[0]).strip()
    if len(out) >= 2 and out[0] in "\"'“‘«" and out[-1] in "\"'”’»":
        out = out[1:-1].strip()
    return out


def stratified_subset(rows, n, seed=0):
    """n rows, spread evenly over dialects (used for validation-during-training and the Groq subset)."""
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["dialect"]].append(r)
    rng = random.Random(seed)
    per = max(1, n // len(by_d))
    out = []
    for d in sorted(by_d):
        part = by_d[d][:]
        rng.shuffle(part)
        out.extend(part[:per])
    return out
