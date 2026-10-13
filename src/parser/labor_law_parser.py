"""Parse a source snapshot without losing clause introductions or Vietnamese points."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class LegalChunk:
    chunk_id: str
    article_number: str
    article_title: str
    clause_number: str | None
    content: str
    embed_text: str
    metadata: dict


class LaborLawParser:
    LAW_NAME = "Bộ luật Lao động"
    LAW_YEAR = 2019
    LAW_DOMAIN = "labor"
    ARTICLE_PATTERN = re.compile(r"^\s*Điều\s+(\d+)\.\s*([^\n]*)", re.MULTILINE)
    CLAUSE_PATTERN = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
    POINT_PATTERN = re.compile(r"^\s*([a-zđ])\)\s+", re.MULTILINE)
    HEADING_PATTERN = re.compile(
        r"^\s*(?:Chương\s+[IVXLCDM]+|Mục\s+\d+)(?:\s|$)", re.MULTILINE
    )

    def __init__(self, source_url: str | None = None):
        self.source_url = source_url

    def parse_html(self, html: str) -> list[LegalChunk]:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("div", class_="content1")
        if main is None:
            raise ValueError(
                "Không tìm thấy nội dung luật (content1); cần kiểm tra cấu trúc nguồn."
            )
        paragraphs = main.find_all("p")
        text = (
            "\n".join(" ".join(p.get_text().split()) for p in paragraphs)
            if paragraphs
            else main.get_text()
        )
        return self.parse_text(
            text, source_hash=hashlib.sha256(html.encode()).hexdigest()
        )

    @staticmethod
    def _outside_quotes(pattern, text):
        # Amendments quote other laws with their own numbered clauses/points.
        # These nested markers must remain part of the outer provision.
        spans = [(m.start(), m.end()) for m in re.finditer("“.*?”", text, re.DOTALL)]
        return [
            m
            for m in pattern.finditer(text)
            if not any(a <= m.start() < b for a, b in spans)
        ]

    def parse_text(self, text: str, source_hash: str | None = None) -> list[LegalChunk]:
        text = text.replace("\r\n", "\n").replace("\xa0", " ")
        source_hash = source_hash or hashlib.sha256(text.encode()).hexdigest()
        version = source_hash[:16]
        timestamp = datetime.now(timezone.utc).isoformat()
        matches = self._outside_quotes(self.ARTICLE_PATTERN, text)
        chunks = []
        for i, match in enumerate(matches):
            number, title = match.group(1), match.group(2).strip()
            body = text[
                match.end() : (
                    matches[i + 1].start() if i + 1 < len(matches) else len(text)
                )
            ]
            heading = self.HEADING_PATTERN.search(body)
            if heading:
                body = body[: heading.start()]
            body = re.split(
                r"\n\s*(?:CHỦ TỊCH QUỐC HỘI|Bộ luật này được Quốc hội)",
                body,
                maxsplit=1,
            )[0].strip()
            chapter_matches = list(
                re.finditer(
                    r"^\s*Chương\s+([IVXLCDM]+)\s*$",
                    text[: match.start()],
                    re.MULTILINE,
                )
            )
            chapter = (
                f"Chương {chapter_matches[-1].group(1)}" if chapter_matches else None
            )
            article_id = f"labor_2019:{version}:dieu_{number}"
            clauses = self._outside_quotes(self.CLAUSE_PATTERN, body)
            article_intro = body[: clauses[0].start()].strip() if clauses else ""
            units = []
            if not clauses:
                units.append((None, None, body, body, "", article_id))
            for j, clause in enumerate(clauses):
                clause_number = clause.group(1)
                clause_body = body[
                    clause.end() : (
                        clauses[j + 1].start() if j + 1 < len(clauses) else len(body)
                    )
                ].strip()
                points = self._outside_quotes(self.POINT_PATTERN, clause_body)
                clause_id = f"{article_id}:khoan_{clause_number}"
                intro = clause_body[: points[0].start()].strip() if points else ""
                if not points:
                    units.append(
                        (clause_number, None, clause_body, clause_body, "", article_id)
                    )
                for k, point in enumerate(points):
                    original = clause_body[
                        point.end() : (
                            points[k + 1].start()
                            if k + 1 < len(points)
                            else len(clause_body)
                        )
                    ].strip()
                    units.append(
                        (
                            clause_number,
                            point.group(1),
                            original,
                            clause_body,
                            intro,
                            clause_id,
                        )
                    )
            for clause, point, original, parent_content, intro, parent_id in units:
                chunk_id = (
                    article_id
                    + (f":khoan_{clause}" if clause else "")
                    + (f":diem_{point}" if point else "")
                )
                content = "\n".join(
                    x
                    for x in [
                        article_intro,
                        intro,
                        (f"{point}) " if point else "") + original,
                    ]
                    if x
                )
                metadata = {
                    "document_id": "labor_2019",
                    "document_version": version,
                    "law_name": self.LAW_NAME,
                    "law_year": self.LAW_YEAR,
                    "domain": self.LAW_DOMAIN,
                    "chapter": chapter,
                    "article": f"Điều {number}",
                    "article_title": title,
                    "clause": clause,
                    "point": point,
                    "chunk_id": chunk_id,
                    "parent_id": parent_id if chunk_id != parent_id else None,
                    "parent_content": "\n".join(
                        x for x in [article_intro, parent_content] if x
                    ),
                    "original_content": original,
                    "source_url": self.source_url,
                    "source_hash": source_hash,
                    "ingested_at": timestamp,
                    "effective_from": None,
                    "effective_to": None,
                    "validity_verified": False,
                }
                label = f"{self.LAW_NAME} {self.LAW_YEAR} — Điều {number}. {title}"
                if clause:
                    label += f" — Khoản {clause}"
                if point:
                    label += f" — Điểm {point}"
                chunks.append(
                    LegalChunk(
                        chunk_id,
                        number,
                        title,
                        clause,
                        content,
                        label + "\n" + content,
                        metadata,
                    )
                )
        if not chunks:
            raise ValueError("Không tìm thấy điều luật trong nguồn.")
        return chunks

    def save_json(self, chunks: list[LegalChunk], output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
