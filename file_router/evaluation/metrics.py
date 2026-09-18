"""Dependency-light answer metrics for short-form multimodal QA."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable, List


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    # Treat western and Indian thousands separators as formatting, so
    # 30,216,492 and 3,02,16,492 compare with 30216492.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s.-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Word-tokenize Latin text and character-tokenize CJK text."""
    normalized = normalize_answer(text)
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+(?:[.-][a-z0-9]+)*",
                      normalized, flags=re.IGNORECASE)


def _overlap_counts(pred_tokens: List[str], gold_tokens: List[str]) -> int:
    return sum((Counter(pred_tokens) & Counter(gold_tokens)).values())


def _token_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    pred, gold = tokenize(prediction), tokenize(reference)
    if not pred or not gold:
        equal = float(pred == gold)
        return equal, equal, equal
    overlap = _overlap_counts(pred, gold)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return precision, recall, f1


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[index:index + n])
                   for index in range(max(0, len(tokens) - n + 1)))


def bleu(prediction: str, reference: str, max_n: int = 1) -> float:
    """Sentence BLEU-N with modified precision and light smoothing."""
    pred, gold = tokenize(prediction), tokenize(reference)
    if not pred or not gold:
        return float(pred == gold)
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams, gold_ngrams = _ngrams(pred, n), _ngrams(gold, n)
        total = sum(pred_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ngrams & gold_ngrams).values())
        precisions.append((overlap + 1.0) / (total + 1.0))
    if any(value <= 0 for value in precisions):
        return 0.0
    brevity_penalty = (1.0 if len(pred) >= len(gold)
                       else math.exp(1.0 - len(gold) / max(len(pred), 1)))
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_n)


def _lcs_length(left: List[str], right: List[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, gold = tokenize(prediction), tokenize(reference)
    if not pred or not gold:
        return float(pred == gold)
    lcs = _lcs_length(pred, gold)
    precision, recall = lcs / len(pred), lcs / len(gold)
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def _numbers(text: str) -> List[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text or "")
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    normalized = []
    for value in values:
        try:
            normalized.append(f"{float(value):g}")
        except ValueError:
            normalized.append(value)
    return normalized


def _char_f1(prediction: str, reference: str) -> float:
    pred = list(re.sub(r"\s+", "", normalize_answer(prediction)))
    gold = list(re.sub(r"\s+", "", normalize_answer(reference)))
    if not pred or not gold:
        return float(pred == gold)
    overlap = _overlap_counts(pred, gold)
    precision, recall = overlap / len(pred), overlap / len(gold)
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def answer_metrics(prediction: str, reference: str) -> dict:
    """Compute all deterministic metrics for one answer pair."""
    norm_pred, norm_gold = normalize_answer(prediction), normalize_answer(reference)
    precision, recall, f1 = _token_prf(prediction, reference)
    gold_numbers = _numbers(reference)
    pred_numbers = _numbers(prediction)
    return {
        "exact_match": float(norm_pred == norm_gold),
        "contains_answer": float(bool(norm_pred and norm_gold) and
                                 (norm_pred in norm_gold or norm_gold in norm_pred)),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "bleu_1": bleu(prediction, reference, max_n=1),
        "bleu_2": bleu(prediction, reference, max_n=2),
        "rouge_l": rouge_l(prediction, reference),
        "char_f1": _char_f1(prediction, reference),
        "edit_similarity": SequenceMatcher(None, norm_pred, norm_gold).ratio(),
        "numeric_exact": (float(pred_numbers == gold_numbers)
                          if gold_numbers else None),
    }


def mean_present(rows: Iterable[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None
