"""Dataset loading and prompt formatting.

The task is real: banking77 is 13k genuine customer-support messages sent to a
retail bank, each labelled with one of 77 intents. We ask the model to emit
strict JSON rather than a bare label, because "did the model produce
machine-parseable output" is the failure mode that actually breaks production
pipelines, and it gives us a second metric that moves sharply with tuning.

Every function here is pure and importable without torch or datasets, so the
formatting contract is unit-testable anywhere.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = (
    "You are an intent classifier for a retail bank's customer support queue.\n"
    "Classify the customer message into exactly one intent from the allowed list.\n"
    # Braces are doubled: this template is rendered with str.format, so a literal
    # brace must be escaped or format() reads it as a replacement field.
    'Respond with ONLY a JSON object of the form {{"intent": "<label>"}}.\n'
    "No prose, no markdown, no explanation.\n\n"
    "Allowed intents:\n{labels}"
)

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def build_system_prompt(labels: list[str]) -> str:
    """Render the system prompt with the label space inlined.

    The label list is given to the baseline model as well as the tuned one, so
    the before/after comparison measures formatting and discrimination rather
    than simply whether the model could have guessed the vocabulary.
    """
    return SYSTEM_PROMPT.format(labels="\n".join(f"- {label}" for label in labels))


def build_messages(text: str, labels: list[str]) -> list[dict[str, str]]:
    """Chat-format one example as the model will see it at inference time."""
    return [
        {"role": "system", "content": build_system_prompt(labels)},
        {"role": "user", "content": text.strip()},
    ]


def target_completion(label: str) -> str:
    """The exact string the model is trained to produce."""
    return json.dumps({"intent": label}, separators=(",", ":"))


def build_training_record(text: str, label: str, labels: list[str]) -> dict[str, Any]:
    """One SFT record: prompt messages plus the assistant turn to learn."""
    return {
        "messages": [
            *build_messages(text, labels),
            {"role": "assistant", "content": target_completion(label)},
        ]
    }


def parse_prediction(raw: str) -> str | None:
    """Extract the predicted intent from raw model output.

    Returns None when the output is not parseable as a JSON object carrying a
    string "intent" key. Tolerates markdown fences and trailing chatter, since
    an untuned model produces both; does NOT tolerate a missing or non-string
    intent, because that is a genuine failure.
    """
    if not raw:
        return None

    candidate = _FENCE_RE.sub("", raw).strip()

    for attempt in (candidate, *(m.group(0) for m in _JSON_OBJECT_RE.finditer(candidate))):
        try:
            parsed = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            intent = parsed.get("intent")
            if isinstance(intent, str) and intent.strip():
                return intent.strip()
    return None


def load_splits(
    dataset_id: str,
    train_rows: int,
    eval_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Download the dataset and return (train, eval, label_names).

    Imported lazily so that `preflight` and the unit tests do not require the
    full ML stack. Row counts are capped rather than exact: a split smaller than
    the requested cap yields whatever it has.
    """
    from datasets import load_dataset

    raw = load_dataset(dataset_id)
    label_names: list[str] = list(raw["train"].features["label"].names)

    train = raw["train"].shuffle(seed=seed).select(range(min(train_rows, len(raw["train"]))))
    eval_key = "test" if "test" in raw else "train"
    eval_pool = raw[eval_key].shuffle(seed=seed + 1)
    evaluation = eval_pool.select(range(min(eval_rows, len(eval_pool))))

    train_records = [
        build_training_record(row["text"], label_names[row["label"]], label_names) for row in train
    ]
    eval_records = [
        {
            "text": row["text"],
            "label": label_names[row["label"]],
            "messages": build_messages(row["text"], label_names),
        }
        for row in evaluation
    ]
    return train_records, eval_records, label_names
