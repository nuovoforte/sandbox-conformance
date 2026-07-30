"""Scoring: JSON-validity rate and intent accuracy.

Two metrics rather than one, deliberately. Accuracy alone hides the difference
between "the model was wrong" and "the model returned something no downstream
service can parse" — and the second is the one that pages an on-call engineer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .lora_data import parse_prediction

if TYPE_CHECKING:
    from .lora_config import RunConfig


@dataclass(frozen=True, slots=True)
class Scores:
    n: int
    json_valid: int
    correct: int
    in_label_space: int

    @property
    def json_valid_rate(self) -> float:
        return self.json_valid / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        """Strict: unparseable output counts as wrong."""
        return self.correct / self.n if self.n else 0.0

    @property
    def accuracy_on_parsed(self) -> float:
        return self.correct / self.json_valid if self.json_valid else 0.0

    @property
    def in_label_space_rate(self) -> float:
        return self.in_label_space / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "json_valid": self.json_valid,
            "correct": self.correct,
            "json_valid_rate": round(self.json_valid_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "accuracy_on_parsed": round(self.accuracy_on_parsed, 4),
            "in_label_space_rate": round(self.in_label_space_rate, 4),
        }


def score(predictions: list[str], gold: list[str], label_space: list[str]) -> Scores:
    """Pure scoring over raw model outputs. No torch, fully unit-testable."""
    if len(predictions) != len(gold):
        raise ValueError(f"length mismatch: {len(predictions)} predictions vs {len(gold)} gold")

    allowed = set(label_space)
    json_valid = correct = in_space = 0
    for raw, truth in zip(predictions, gold, strict=True):
        parsed = parse_prediction(raw)
        if parsed is None:
            continue
        json_valid += 1
        if parsed in allowed:
            in_space += 1
        if parsed == truth:
            correct += 1
    return Scores(n=len(gold), json_valid=json_valid, correct=correct, in_label_space=in_space)


def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    cfg: RunConfig,
    batch_size: int = 16,
) -> list[str]:
    """Batched greedy generation over evaluation records."""
    import torch

    outputs: list[str] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice off the prompt so we score only the completion.
        prompt_len = batch["input_ids"].shape[1]
        for row in generated[:, prompt_len:]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True))

    return outputs
