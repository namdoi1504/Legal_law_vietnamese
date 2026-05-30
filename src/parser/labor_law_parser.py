# src/parser/labor_law_parser.py
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional
from bs4 import BeautifulSoup


@dataclass
class LegalChunk:
    chunk_id: str
    article_number: str
    article_title: str
    clause_number: Optional[str]
    content: str
    embed_text: str
    metadata: dict


class LaborLawParser:
    LAW_NAME = "Bộ luật Lao động"
    LAW_YEAR = 2019
    LAW_DOMAIN = "labor"

    ARTICLE_PATTERN = re.compile(
        r"Điều\s+(\d+)\.\s*(.*?)(?=\nĐiều\s+\d+\.|$)",
        re.DOTALL
    )
    CLAUSE_PATTERN = re.compile(
        r"(\d+)\.\s+(.*?)(?=\n\d+\.\s+|\Z)",
        re.DOTALL
    )
    POINT_PATTERN = re.compile(
        r"([a-z]\))\s+(.*?)(?=\n[a-z]\)\s+|\Z)",
        re.DOTALL
    )
    CHAPTER_PATTERN = re.compile(
        r"Chương\s+([\w]+)\s*\n(.*?)(?=\nChương|\Z)",
        re.DOTALL
    )

    def parse_html(self, html: str) -> list[LegalChunk]:
        soup = BeautifulSoup(html, "html.parser")
        main_content = soup.find("div", class_="content1")
        if not main_content:
            main_content = soup.get_text()
        else:
            main_content = main_content.get_text()
        return self.parse_text(self._clean_raw_text(main_content))

    def _clean_raw_text(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def parse_text(self, text: str) -> list[LegalChunk]:
        chunks = []
        current_chapter = "Chưa xác định"

        for match in self.ARTICLE_PATTERN.finditer(text):
            article_num = match.group(1)
            article_title = match.group(2).split("\n")[0].strip()
            article_body = match.group(2)

            pos = match.start()
            chapter_match = self._find_chapter_at_pos(text, pos)
            if chapter_match:
                current_chapter = chapter_match

            # Luôn thử tách theo Khoản
            clause_chunks = self._split_by_clause(
                article_num, article_title,
                article_body, current_chapter
            )

            if len(clause_chunks) > 1:
                chunks.extend(clause_chunks)
            else:
                chunks.append(self._make_chunk(
                    article_num, article_title,
                    article_body.strip(), current_chapter
                ))

        return chunks

    def _split_by_clause(
        self, article_num, article_title,
        body, chapter
    ) -> list[LegalChunk]:
        chunks = []
        clauses = self.CLAUSE_PATTERN.findall(body)

        if not clauses:
            return [self._make_chunk(article_num, article_title, body, chapter)]

        for clause_num, clause_content in clauses:
            # Thử tách theo Điểm (a, b, c) bên trong Khoản
            points = self.POINT_PATTERN.findall(clause_content)

            if len(points) > 1:
                # Có nhiều điểm → tách theo Điểm
                for point_label, point_content in points:
                    chunk = self._make_chunk(
                        article_num, article_title,
                        point_content.strip(), chapter,
                        clause_number=f"{clause_num}{point_label}"
                    )
                    chunks.append(chunk)
            else:
                # Không có điểm → giữ nguyên Khoản
                chunk = self._make_chunk(
                    article_num, article_title,
                    clause_content.strip(), chapter,
                    clause_number=clause_num
                )
                chunks.append(chunk)

        return chunks

    def _make_chunk(
        self, article_num, article_title,
        content, chapter, clause_number=None
    ) -> LegalChunk:
        chunk_id = f"labor_2019_dieu_{article_num}"
        if clause_number:
            chunk_id += f"_khoan_{clause_number}"

        if clause_number:
            embed_text = (
                f"{self.LAW_NAME} {self.LAW_YEAR} - {chapter}\n"
                f"Điều {article_num}. {article_title}\n"
                f"Khoản {clause_number}: {content}"
            )
        else:
            embed_text = (
                f"{self.LAW_NAME} {self.LAW_YEAR} - {chapter}\n"
                f"Điều {article_num}. {article_title}\n"
                f"{content}"
            )

        return LegalChunk(
            chunk_id=chunk_id,
            article_number=article_num,
            article_title=article_title,
            clause_number=clause_number,
            content=content,
            embed_text=embed_text,
            metadata={
                "law_name": self.LAW_NAME,
                "law_year": self.LAW_YEAR,
                "domain": self.LAW_DOMAIN,
                "chapter": chapter,
                "article": f"Điều {article_num}",
                "article_title": article_title,
                "clause": clause_number,
                "chunk_id": chunk_id,
            }
        )

    def _find_chapter_at_pos(self, text, pos):
        candidates = list(self.CHAPTER_PATTERN.finditer(text))
        current = None
        for m in candidates:
            if m.start() <= pos:
                current = f"Chương {m.group(1)} - {m.group(2).strip()[:50]}"
        return current

    def save_json(self, chunks: list[LegalChunk], output_path: str):
        data = [asdict(c) for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} chunks to {output_path}")