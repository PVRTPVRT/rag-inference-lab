"""Claim-to-citation natural-language-inference verification."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
CITATION_BLOCK_RE = re.compile(r"\[S\d+\s*(?:,\s*S\d+\s*)*\]")
SOURCE_ID_RE = re.compile(r"S(\d+)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_verifier = None
WORD_RE = re.compile(r"[a-z0-9]+")
FOCUS_STOPWORDS = {
    "and", "are", "but", "can", "does", "for", "from", "has", "have",
    "how", "into", "its", "not", "that", "the", "their", "then", "this",
    "through", "to", "use", "uses", "using", "was", "were", "what", "when",
    "where", "which", "while", "with",
}


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in WORD_RE.findall(text.lower())
        if len(word) > 2 and word not in FOCUS_STOPWORDS
    }


def focus_premise(premise: str, hypothesis: str, max_sentences: int = 3) -> str:
    """Keep the most lexically relevant evidence sentences in document order."""
    if max_sentences <= 0:
        return premise
    sentences = [item.strip() for item in SENTENCE_RE.split(premise) if item.strip()]
    if len(sentences) <= max_sentences:
        return premise
    hypothesis_words = _content_words(hypothesis)
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (
            len(_content_words(sentences[index]) & hypothesis_words),
            -index,
        ),
        reverse=True,
    )
    selected = ranked[:max_sentences]
    if not selected or not any(
        _content_words(sentences[index]) & hypothesis_words for index in selected
    ):
        return premise
    return " ".join(sentences[index] for index in sorted(selected))


@dataclass(frozen=True)
class NLIResult:
    predicted_label: str
    probabilities: dict[str, float]
    latency_ms: float


class EntailmentVerifier:
    """Lazy local DeBERTa NLI classifier with explicit label validation."""

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        device: str | None = None,
        focus_sentences: int | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = (device or os.getenv("RAG_NLI_DEVICE", "cpu")).strip() or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        if focus_sentences is None:
            focus_sentences = int(os.getenv("RAG_NLI_FOCUS_SENTENCES", "3"))
        if focus_sentences < 0:
            raise ValueError("focus_sentences must be non-negative")
        self.focus_sentences = focus_sentences
        self.model.eval()
        labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        expected = {"contradiction", "entailment", "neutral"}
        if set(labels.values()) != expected:
            raise RuntimeError(f"unexpected NLI label mapping: {labels}")
        self.labels = labels
        self.model_name = model_name

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        focused_premise = focus_premise(premise, hypothesis, self.focus_sentences)
        started = time.perf_counter()
        features = self.tokenizer(
            focused_premise,
            hypothesis,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )
        features = {name: value.to(self.device) for name, value in features.items()}
        with self.torch.inference_mode():
            logits = self.model(**features).logits[0]
            values = self.torch.softmax(logits.float(), dim=-1).cpu().tolist()
        probabilities = {
            self.labels[index]: round(float(value), 6)
            for index, value in enumerate(values)
        }
        predicted = max(probabilities, key=probabilities.get)
        return NLIResult(
            predicted_label=predicted,
            probabilities=probabilities,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def get_verifier() -> EntailmentVerifier:
    global _verifier
    if _verifier is None:
        _verifier = EntailmentVerifier()
    return _verifier


def extract_citation_ids(text: str) -> list[int]:
    """Return IDs from both `[S1][S2]` and `[S1, S2]` citation forms."""
    return [
        int(source_id) for block in CITATION_BLOCK_RE.findall(text)
        for source_id in SOURCE_ID_RE.findall(block)
    ]


def extract_claims(answer: str) -> list[dict]:
    """Split an answer into sentence-level claims and retain attached source IDs."""
    claims = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        if not line:
            continue
        for sentence in SENTENCE_RE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            citation_ids = extract_citation_ids(sentence)
            claim = CITATION_BLOCK_RE.sub("", sentence).strip()
            claim = re.sub(r"\s+([.,!?;:])", r"\1", claim)
            if claim:
                claims.append({"claim": claim, "citation_ids": citation_ids})
    return claims


def verify_cited_answer(
    answer: str,
    chunks: list[dict],
    *,
    verifier: EntailmentVerifier | None = None,
    entailment_threshold: float = 0.5,
    contradiction_threshold: float = 0.5,
) -> dict:
    """Verify every cited claim against its cited chunks without inventing citations."""
    verifier = verifier or get_verifier()
    checked = []
    for item in extract_claims(answer):
        valid_ids = [
            source_id for source_id in item["citation_ids"]
            if 1 <= source_id <= len(chunks)
        ]
        invalid_ids = [
            source_id for source_id in item["citation_ids"]
            if source_id not in valid_ids
        ]
        evidence = []
        for source_id in valid_ids:
            result = verifier.predict(chunks[source_id - 1]["text"], item["claim"])
            evidence.append({
                "source_id": f"S{source_id}",
                "predicted_label": result.predicted_label,
                "probabilities": result.probabilities,
                "latency_ms": result.latency_ms,
            })
        supported = any(
            row["predicted_label"] == "entailment"
            and row["probabilities"]["entailment"] >= entailment_threshold
            for row in evidence
        )
        contradicted = any(
            row["predicted_label"] == "contradiction"
            and row["probabilities"]["contradiction"] >= contradiction_threshold
            for row in evidence
        )
        checked.append({
            **item,
            "invalid_citation_ids": invalid_ids,
            "supported": supported,
            "contradicted": contradicted,
            "evidence": evidence,
        })
    factual = [item for item in checked if item["claim"]]
    cited = [item for item in factual if item["citation_ids"]]
    supported = [item for item in factual if item["supported"]]
    contradicted = [item for item in factual if item["contradicted"]]
    return {
        "claims": checked,
        "summary": {
            "claim_count": len(factual),
            "cited_claim_count": len(cited),
            "citation_coverage": round(len(cited) / len(factual), 4) if factual else None,
            "supported_claim_count": len(supported),
            "entailment_rate": round(len(supported) / len(factual), 4) if factual else None,
            "contradicted_claim_count": len(contradicted),
            "contradiction_rate": round(len(contradicted) / len(factual), 4) if factual else None,
            "invalid_citation_count": sum(len(item["invalid_citation_ids"]) for item in factual),
        },
    }
