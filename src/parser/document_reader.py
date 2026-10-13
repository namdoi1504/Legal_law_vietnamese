from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from configs.settings import settings


class DocumentReader:
    def read_bytes(self, file_bytes, filename):
        if not file_bytes or len(file_bytes) > settings.MAX_UPLOAD_BYTES:
            raise ValueError("File rỗng hoặc vượt giới hạn kích thước.")
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            if not file_bytes.startswith(b"%PDF-"):
                raise ValueError("Nội dung file không phải PDF.")
            import fitz

            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if doc.is_encrypted:
                    raise ValueError("PDF có mật khẩu; cần bản có thể đọc.")
                if len(doc) > 200:
                    raise ValueError("PDF vượt giới hạn 200 trang.")
                pages = [page.get_text().strip() for page in doc]
                unreadable = any(not text and (page.get_images() or page.get_drawings())
                                 for page,text in zip(doc,pages))
            if not any(pages) or unreadable:
                raise ValueError(
                    "PDF có trang không đọc được text; cần OCR trước khi rà soát đầy đủ."
                )
            text = "\n\n".join(
                f"[Trang {i}]\n{page}" for i, page in enumerate(pages, 1)
            )
        elif suffix == ".docx":
            try:
                with ZipFile(BytesIO(file_bytes)) as archive:
                    if "word/document.xml" not in archive.namelist():
                        raise ValueError("File không phải DOCX hợp lệ.")
                    if sum(i.file_size for i in archive.infolist()) > 50 * 1024 * 1024:
                        raise ValueError("Nội dung DOCX sau giải nén quá lớn.")
            except BadZipFile as exc:
                raise ValueError("File DOCX không hợp lệ.") from exc
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            doc = Document(BytesIO(file_bytes))
            blocks = []
            for block in doc.iter_inner_content():
                if isinstance(block, Paragraph):
                    blocks.append(block.text)
                elif isinstance(block, Table):
                    blocks.extend(
                        " | ".join(cell.text for cell in row.cells)
                        for row in block.rows
                    )
            text = "\n\n".join(blocks).strip()
        else:
            raise ValueError(
                "Chỉ hỗ trợ PDF có text và DOCX. Hãy chuyển file .doc sang .docx."
            )
        if not text.strip():
            raise ValueError("Không tìm thấy nội dung văn bản.")
        if len(text) > settings.MAX_DOCUMENT_CHARS:
            raise ValueError("Tài liệu vượt giới hạn nội dung cho một lần kiểm tra.")
        return text
