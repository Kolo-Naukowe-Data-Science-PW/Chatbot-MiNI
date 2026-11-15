"""
Attachment processing for the chat API.

Every file a user attaches is turned into something every LLM can consume:

- documents (PDF, DOCX, XLSX, TXT/MD/CSV/JSON) -> plain text,
- images (JPEG, PNG, WEBP, GIF) -> a re-encoded, downscaled JPEG data URL
  (re-encoding also strips EXIF metadata such as GPS position),
- scanned PDFs without a text layer -> the page images embedded in the PDF.

The file type is detected from the file content (magic bytes), not only from
the extension, so a renamed file is still handled correctly.

All extracted text passes through this module before it reaches the prompt,
which is the single place where prompt anonymization can be applied to
attachment content later on.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import logging
import os
import zipfile
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits (overridable with environment variables)
# ---------------------------------------------------------------------------

MAX_FILE_BYTES = int(os.getenv("ATTACHMENT_MAX_FILE_MB", "10")) * 1024 * 1024
MAX_FILES_PER_MESSAGE = int(os.getenv("ATTACHMENT_MAX_FILES", "5"))
MAX_CHARS_PER_FILE = int(os.getenv("ATTACHMENT_MAX_CHARS_PER_FILE", "40000"))
MAX_TOTAL_CHARS = int(os.getenv("ATTACHMENT_MAX_TOTAL_CHARS", "80000"))
MAX_HISTORY_CHARS_PER_FILE = int(os.getenv("ATTACHMENT_MAX_HISTORY_CHARS", "8000"))
MAX_PDF_PAGES = int(os.getenv("ATTACHMENT_MAX_PDF_PAGES", "100"))
MAX_SCANNED_PDF_IMAGES = int(os.getenv("ATTACHMENT_MAX_SCANNED_PAGES", "5"))
MAX_XLSX_ROWS_PER_SHEET = int(os.getenv("ATTACHMENT_MAX_XLSX_ROWS", "500"))
MAX_IMAGE_SIDE_PX = int(os.getenv("ATTACHMENT_MAX_IMAGE_SIDE", "1600"))
MAX_IMAGE_DATA_URL_CHARS = 8 * 1024 * 1024  # re-encoded images are far smaller

# Model used when the user attaches an image but the selected model cannot see.
VISION_FALLBACK_MODEL = os.getenv("ATTACHMENT_VISION_MODEL", "openai/gpt-4o-mini")

# OpenRouter model id prefixes known to accept image input.
VISION_MODEL_PREFIXES = (
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-5",
    "google/gemini",
    "anthropic/claude",
    "mistralai/mistral-small-3",
    "mistralai/pixtral",
    "meta-llama/llama-3.2-11b-vision",
    "meta-llama/llama-3.2-90b-vision",
    "meta-llama/llama-4",
    "qwen/qwen2.5-vl",
    "qwen/qwen-vl",
)

TEXT_EXTENSIONS = {"txt", "md", "markdown", "csv", "tsv", "json", "log", "xml", "html"}
SUPPORTED_EXTENSIONS = sorted(
    TEXT_EXTENSIONS
    | {"pdf", "docx", "xlsx", "xlsm", "jpg", "jpeg", "png", "webp", "gif"}
)


class AttachmentError(ValueError):
    """Raised when a file cannot be accepted. The message is shown to the user (PL)."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ProcessedAttachment:
    """Result of processing one uploaded file."""

    filename: str
    kind: str  # "pdf" | "docx" | "xlsx" | "text" | "image"
    text: str | None = None
    images: list[str] = field(default_factory=list)  # JPEG data URLs
    truncated: bool = False
    pages: int | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def detect_kind(filename: str, content: bytes) -> str:
    """Return the attachment kind based on magic bytes, falling back to the extension."""
    head = content[:16]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image"
    if head[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile as exc:
            raise AttachmentError(
                "Plik jest uszkodzony (nieprawidłowe archiwum)."
            ) from exc
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        raise AttachmentError(
            "Nieobsługiwany format pliku. Obsługiwane: "
            + ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS),
            status_code=415,
        )
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise AttachmentError(
            "Stare formaty Office (.doc, .xls) nie są obsługiwane. "
            "Zapisz plik jako .docx lub .xlsx i spróbuj ponownie.",
            status_code=415,
        )

    ext = _extension(filename)
    if ext in TEXT_EXTENSIONS:
        return "text"
    if _looks_like_text(content):
        return "text"
    raise AttachmentError(
        "Nieobsługiwany format pliku. Obsługiwane: "
        + ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS),
        status_code=415,
    )


def _looks_like_text(content: bytes) -> bool:
    sample = content[:4096]
    if not sample or b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _image_to_data_url(raw: bytes) -> str:
    """Validate an image, downscale it and re-encode as JPEG (drops EXIF metadata)."""
    from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AttachmentError(
            "Nie udało się odczytać obrazu (plik uszkodzony?)."
        ) from exc
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _extract_pdf(content: bytes) -> tuple[str, int, list[str], str | None]:
    from pypdf import PdfReader  # noqa: PLC0415
    from pypdf.errors import PdfReadError  # noqa: PLC0415

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise AttachmentError("PDF jest zabezpieczony hasłem.")
            except AttachmentError:
                raise
            except Exception as exc:
                raise AttachmentError("PDF jest zabezpieczony hasłem.") from exc
        total_pages = len(reader.pages)
        pages = reader.pages[:MAX_PDF_PAGES]
        parts = []
        for number, page in enumerate(pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one broken page must not kill the file
                logger.warning("Failed to extract text from PDF page %d", number)
                page_text = ""
            if page_text.strip():
                parts.append(f"[Strona {number}]\n{page_text.strip()}")
    except AttachmentError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise AttachmentError(
            "Nie udało się odczytać pliku PDF (plik uszkodzony?)."
        ) from exc

    note = None
    if total_pages > MAX_PDF_PAGES:
        note = f"Odczytano tylko pierwsze {MAX_PDF_PAGES} z {total_pages} stron."

    text = "\n\n".join(parts)
    images: list[str] = []
    # Scanned PDF: almost no text layer -> send embedded page images to the model.
    if len(text.strip()) < 25 * max(1, len(pages)):
        for page in pages:
            if len(images) >= MAX_SCANNED_PDF_IMAGES:
                break
            try:
                page_images = list(page.images)
            except Exception:  # noqa: BLE001
                continue
            if page_images:
                biggest = max(page_images, key=lambda im: len(im.data))
                try:
                    images.append(_image_to_data_url(biggest.data))
                except AttachmentError:
                    continue
        if images:
            note = (
                "PDF nie zawiera warstwy tekstowej (skan) — przesłano obrazy "
                f"pierwszych {len(images)} stron."
            )
    return text, total_pages, images, note


def _extract_docx(content: bytes) -> str:
    from docx import Document  # noqa: PLC0415
    from docx.table import Table  # noqa: PLC0415
    from docx.text.paragraph import Paragraph  # noqa: PLC0415

    try:
        document = Document(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - python-docx raises many types
        raise AttachmentError(
            "Nie udało się odczytać pliku DOCX (plik uszkodzony?)."
        ) from exc

    lines: list[str] = []

    def table_lines(table: Table) -> list[str]:
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                cell_text = " ".join(p.text.strip() for p in cell.paragraphs).strip()
                cells.append(cell_text.replace("|", "/"))
            # merged cells repeat the same text – collapse consecutive duplicates
            deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
            if any(deduped):
                rows.append("| " + " | ".join(deduped) + " |")
        return rows

    # Walk the body in document order so tables stay where they were.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            if style.startswith("heading") or style.startswith("nagłówek"):
                text = f"## {text}"
            elif "list" in style:
                text = f"- {text}"
            lines.append(text)
        elif tag == "tbl":
            lines.extend(table_lines(Table(child, document)))
            lines.append("")

    for section in document.sections:
        for part in (section.header, section.footer):
            try:
                extra = " ".join(
                    p.text.strip() for p in part.paragraphs if p.text.strip()
                )
            except Exception:  # noqa: BLE001
                extra = ""
            if extra and extra not in lines:
                lines.append(extra)

    return "\n".join(lines).strip()


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M").removesuffix(" 00:00")
        except ValueError:
            pass
    return str(value).replace("\n", " ").replace("|", "/").strip()


def _extract_xlsx(content: bytes) -> tuple[str, str | None]:
    from openpyxl import load_workbook  # noqa: PLC0415

    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise AttachmentError(
            "Nie udało się odczytać pliku XLSX (plik uszkodzony?)."
        ) from exc

    blocks: list[str] = []
    cut_sheets: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[list[str]] = []
            total_rows = 0
            for row in sheet.iter_rows(values_only=True):
                cells = [_format_cell(v) for v in row]
                while cells and not cells[-1]:
                    cells.pop()
                if not any(cells):
                    continue
                total_rows += 1
                if len(rows) < MAX_XLSX_ROWS_PER_SHEET:
                    rows.append(cells)
            if not rows:
                continue
            width = max(len(r) for r in rows)
            table = [
                "| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows
            ]
            header = f"### Arkusz: {sheet.title}"
            if total_rows > len(rows):
                cut_sheets.append(sheet.title)
                header += f" (pokazano {len(rows)} z {total_rows} wierszy)"
            blocks.append(header + "\n" + "\n".join(table))
    finally:
        workbook.close()

    note = None
    if cut_sheets:
        note = "Część wierszy pominięto w arkuszach: " + ", ".join(cut_sheets) + "."
    return "\n\n".join(blocks), note


def _csv_to_table(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        return text
    rows = list(csv.reader(io.StringIO(text), dialect))
    if len(rows) < 2:
        return text
    return "\n".join("| " + " | ".join(c.strip() for c in r) + " |" for r in rows if r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n[... treść skrócona ...]", True


def process_file(filename: str, content: bytes) -> ProcessedAttachment:
    """Validate an uploaded file and convert it into prompt-ready content."""
    filename = os.path.basename(filename or "plik").strip() or "plik"
    if not content:
        raise AttachmentError("Plik jest pusty.")
    if len(content) > MAX_FILE_BYTES:
        raise AttachmentError(
            f"Plik jest za duży (maks. {MAX_FILE_BYTES // (1024 * 1024)} MB).",
            status_code=413,
        )

    kind = detect_kind(filename, content)
    result = ProcessedAttachment(filename=filename, kind=kind)

    if kind == "image":
        result.images = [_image_to_data_url(content)]
        return result

    if kind == "pdf":
        text, pages, images, note = _extract_pdf(content)
        result.pages, result.images, result.note = pages, images, note
    elif kind == "docx":
        text = _extract_docx(content)
    elif kind == "xlsx":
        text, result.note = _extract_xlsx(content)
    else:
        text = _decode_text(content)
        if _extension(filename) in ("csv", "tsv"):
            text = _csv_to_table(text)

    text = text.replace("\x00", "").strip()
    if not text and not result.images:
        raise AttachmentError("Nie znaleziono w pliku żadnej treści do odczytania.")

    result.text, result.truncated = _truncate(text, MAX_CHARS_PER_FILE)
    if result.truncated:
        extra = f"Plik jest długi — przekazano pierwsze {MAX_CHARS_PER_FILE} znaków."
        result.note = f"{result.note} {extra}" if result.note else extra
    return result


def process_base64(filename: str, data: str) -> ProcessedAttachment:
    """Process a file sent as base64 (legacy format of the /chat request)."""
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError("Nieprawidłowe dane pliku (base64).") from exc
    return process_file(filename, raw)


def validate_image_data_url(url: str) -> str:
    """Accept only re-encodable image data URLs coming back from the client."""
    if not isinstance(url, str) or not url.startswith("data:image/") or "," not in url:
        raise AttachmentError("Nieprawidłowy obraz w załączniku.")
    if len(url) > MAX_IMAGE_DATA_URL_CHARS:
        raise AttachmentError("Obraz w załączniku jest za duży.", status_code=413)
    if url.startswith("data:image/jpeg;base64,"):
        return url
    # Anything else gets re-encoded so the model always receives a clean JPEG.
    return _image_to_data_url(base64.b64decode(url.split(",", 1)[1]))


def render_attachments_text(
    attachments: list[dict], per_file_limit: int | None = None
) -> str:
    """
    Render attachment text as a clearly delimited block for the prompt.

    Each item is a dict with at least ``filename`` and optionally ``text``,
    ``kind``, ``images`` and ``note``.
    """
    blocks: list[str] = []
    budget = MAX_TOTAL_CHARS
    for att in attachments:
        name = att.get("filename") or "plik"
        text = (att.get("text") or "").strip()
        limit = per_file_limit or MAX_CHARS_PER_FILE
        text, _ = _truncate(text, min(limit, max(budget, 0)))
        budget -= len(text)
        header = f'<<<ZAŁĄCZNIK: "{name}"'
        if att.get("kind"):
            header += f" ({att['kind']})"
        header += ">>>"
        body_parts = []
        if att.get("note"):
            body_parts.append(f"[Uwaga systemu: {att['note']}]")
        if text:
            body_parts.append(text)
        if att.get("images"):
            body_parts.append(
                f"[Do wiadomości dołączono {len(att['images'])} obraz(y) z tego pliku.]"
            )
        if not body_parts:
            body_parts.append("[brak treści]")
        blocks.append(
            header
            + "\n"
            + "\n".join(body_parts)
            + f'\n<<<KONIEC ZAŁĄCZNIKA: "{name}">>>'
        )
    return "\n\n".join(blocks)


def is_vision_model(model: str | None) -> bool:
    return bool(model) and model.startswith(VISION_MODEL_PREFIXES)


def ensure_vision_model(model_config: dict | None, has_images: bool) -> dict | None:
    """
    If images are attached and the chosen model cannot read images,
    switch to ``VISION_FALLBACK_MODEL``. Returns a (possibly new) config dict.
    """
    if not has_images:
        return model_config
    config = dict(model_config or {})
    if not is_vision_model(config.get("model")):
        logger.info(
            "Model %s cannot read images; using %s for this request.",
            config.get("model"),
            VISION_FALLBACK_MODEL,
        )
        config["model"] = VISION_FALLBACK_MODEL
    return config
