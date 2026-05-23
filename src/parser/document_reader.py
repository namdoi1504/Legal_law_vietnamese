# src/parser/document_reader.py
import fitz  # pymupdf
from docx import Document
from pathlib import Path
import tempfile
import os


class DocumentReader:

    def read_bytes(self, file_bytes: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                return self._read_pdf(tmp_path)
            elif suffix in [".docx", ".doc"]:
                return self._read_docx(tmp_path)
            else:
                raise ValueError(f"Không hỗ trợ định dạng: {suffix}")
        finally:
            os.unlink(tmp_path)

    def _read_pdf(self, path: str) -> str:
        doc = fitz.open(path)
        texts = [page.get_text() for page in doc]
        doc.close()
        full = "\n".join(texts).strip()
        if not full:
            raise ValueError("PDF không có text — có thể là file scan.")
        return full

    def _read_docx(self, path: str) -> str:
        doc = Document(path)
        texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(texts)