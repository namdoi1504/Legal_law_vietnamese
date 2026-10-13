"""Validate source identity and verbatim evidence; semantic support is evaluated separately."""

import json
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from src.pipeline.context import source_label


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=12, max_length=5000)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    evidence: list[Evidence] = Field(min_length=1, max_length=8)


class AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "partial", "needs_information", "insufficient_evidence"]
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=8)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_answer(raw, chunks):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    payload = AnswerPayload.model_validate(json.loads(raw))
    sources = {c["source_id"]: c for c in chunks}
    if payload.status in ("answered", "partial") and not payload.claims:
        raise ValueError("Answer has no supported claims")
    if payload.status == "insufficient_evidence" and payload.claims:
        raise ValueError("Insufficient evidence cannot contain legal conclusions")
    if (
        payload.status in ("needs_information", "partial")
        and not payload.missing_information
    ):
        raise ValueError("Missing-information status requires a question or limitation")
    for claim in payload.claims:
        if re.search(r"\[S\d+\]", claim.text):
            raise ValueError(
                "Source markers must be rendered from evidence, not generated in claim text"
            )
        for evidence in claim.evidence:
            source = sources.get(evidence.source_id)
            if source is None or normalize(evidence.quote) not in normalize(
                source["content"]
            ):
                raise ValueError("Source ID or evidence quotation is invalid")
    return payload


def render_answer(payload, chunks):
    sources = {c["source_id"]: c for c in chunks}
    used = list(
        dict.fromkeys(e.source_id for claim in payload.claims for e in claim.evidence)
    )
    text = "\n\n".join(
        claim.text
        + " "
        + "".join(
            f"[{sid}]" for sid in dict.fromkeys(e.source_id for e in claim.evidence)
        )
        for claim in payload.claims
    )
    if payload.missing_information:
        text += "\n\nThông tin cần bổ sung / giới hạn:\n" + "\n".join(
            "- " + x for x in payload.missing_information
        )
    if not text.strip():
        text = "Tôi chưa tìm thấy đủ căn cứ trong dữ liệu hiện có để trả lời."
    citations = [
        f'[{sid}] {source_label(sources[sid]["metadata"])}'
        + (
            f' — {sources[sid]["metadata"]["source_url"]}'
            if sources[sid]["metadata"].get("source_url")
            else ""
        )
        for sid in used
    ]
    return text.strip(), citations, [sources[sid] for sid in used]
