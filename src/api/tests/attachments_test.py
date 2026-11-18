"""Tests for src/api/attachments.py and attachment handling in the prompt builder."""

import base64
import io

import pytest
from PIL import Image

from src.api import attachments as att
from src.api.models import Attachment, Message
from src.api.prompt_builder import build_messages

# ---------------------------------------------------------------------------
# Fixture builders (no binary files in the repo)
# ---------------------------------------------------------------------------


def make_text_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def make_image(fmt: str = "PNG", size=(40, 30), mode="RGB") -> bytes:
    img = Image.new(mode, size, (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_scanned_pdf() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), (255, 255, 255)).save(buf, format="PDF")
    return buf.getvalue()


def make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("Regulamin praktyk", level=1)
    doc.add_paragraph("Praktyki trwają cztery tygodnie.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Przedmiot"
    table.cell(0, 1).text = "ECTS"
    table.cell(1, 0).text = "Analiza"
    table.cell(1, 1).text = "5"
    doc.add_paragraph("Po tabeli.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_xlsx(rows: int = 3) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Oceny"
    ws.append(["Imię", "Ocena"])
    for i in range(rows):
        ws.append([f"Student {i}", 4.0])
    ws2 = wb.create_sheet("Pusty")
    ws2["A1"] = None
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,expected",
    [
        (lambda: make_text_pdf(["Hello"]), "pdf"),
        (lambda: make_image("PNG"), "image"),
        (lambda: make_image("JPEG"), "image"),
        (lambda: make_image("GIF"), "image"),
        (lambda: make_image("WEBP"), "image"),
        (make_docx, "docx"),
        (make_xlsx, "xlsx"),
    ],
)
def test_detect_kind_uses_content_not_extension(builder, expected):
    # deliberately wrong extension
    assert att.detect_kind("plik.bin", builder()) == expected


def test_detect_kind_text_by_extension_and_content():
    assert att.detect_kind("notatki.md", b"# tytul") == "text"
    assert att.detect_kind("bez_rozszerzenia", "zażółć gęślą".encode()) == "text"


def test_detect_kind_rejects_legacy_office_and_binary():
    with pytest.raises(att.AttachmentError) as exc:
        att.detect_kind("stary.doc", b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
    assert exc.value.status_code == 415
    with pytest.raises(att.AttachmentError):
        att.detect_kind("program.exe", b"MZ\x00\x00binary")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_pdf_text_extraction_with_page_markers():
    result = att.process_file(
        "plan.pdf", make_text_pdf(["Pierwsza strona", "Druga strona"])
    )
    assert result.kind == "pdf"
    assert result.pages == 2
    assert "[Strona 1]" in result.text and "Pierwsza strona" in result.text
    assert "[Strona 2]" in result.text and "Druga strona" in result.text
    assert result.images == []


def test_scanned_pdf_falls_back_to_page_images():
    result = att.process_file("skan.pdf", make_scanned_pdf())
    assert result.kind == "pdf"
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/jpeg;base64,")
    assert "skan" in result.note


def test_broken_pdf_gives_user_message():
    with pytest.raises(att.AttachmentError) as exc:
        att.process_file("zly.pdf", b"%PDF-1.4\nto nie jest pdf")
    assert "PDF" in str(exc.value)


def test_docx_extraction_keeps_headings_tables_and_order():
    result = att.process_file("regulamin.docx", make_docx())
    assert result.kind == "docx"
    text = result.text
    assert "## Regulamin praktyk" in text
    assert "| Przedmiot | ECTS |" in text
    assert "| Analiza | 5 |" in text
    assert (
        text.index("cztery tygodnie")
        < text.index("Przedmiot")
        < text.index("Po tabeli")
    )


def test_xlsx_extraction_as_tables():
    result = att.process_file("oceny.xlsx", make_xlsx())
    assert result.kind == "xlsx"
    assert "### Arkusz: Oceny" in result.text
    assert "| Imię | Ocena |" in result.text
    assert "| Student 0 | 4 |" in result.text  # 4.0 rendered as 4
    assert "Pusty" not in result.text  # empty sheets are skipped


def test_xlsx_row_limit(monkeypatch):
    monkeypatch.setattr(att, "MAX_XLSX_ROWS_PER_SHEET", 5)
    result = att.process_file("duzy.xlsx", make_xlsx(rows=20))
    assert "pokazano 5 z 21 wierszy" in result.text
    assert result.note and "Oceny" in result.note


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF"])
def test_images_are_reencoded_as_jpeg(fmt):
    result = att.process_file(f"foto.{fmt.lower()}", make_image(fmt))
    assert result.kind == "image"
    assert result.text is None
    assert result.images[0].startswith("data:image/jpeg;base64,")


def test_transparent_png_and_downscaling(monkeypatch):
    monkeypatch.setattr(att, "MAX_IMAGE_SIDE_PX", 100)
    raw = make_image("PNG", size=(1000, 500), mode="RGBA")
    url = att.process_file("logo.png", raw).images[0]
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.format == "JPEG"
    assert max(decoded.size) == 100


def test_text_files_and_encodings():
    assert att.process_file("a.txt", "Zażółć".encode("cp1250")).text == "Zażółć"
    assert att.process_file("b.txt", "﻿BOM".encode()).text == "BOM"


def test_csv_becomes_table():
    result = att.process_file("dane.csv", b"a;b\n1;2\n")
    assert "| a | b |" in result.text and "| 1 | 2 |" in result.text


def test_empty_too_large_and_contentless_files(monkeypatch):
    with pytest.raises(att.AttachmentError):
        att.process_file("pusty.txt", b"")
    with pytest.raises(att.AttachmentError):
        att.process_file("spacje.txt", b"   \n  ")
    monkeypatch.setattr(att, "MAX_FILE_BYTES", 10)
    with pytest.raises(att.AttachmentError) as exc:
        att.process_file("duzy.txt", b"x" * 11)
    assert exc.value.status_code == 413


def test_long_text_is_truncated(monkeypatch):
    monkeypatch.setattr(att, "MAX_CHARS_PER_FILE", 50)
    result = att.process_file("dlugi.txt", b"a" * 500)
    assert result.truncated
    assert "treść skrócona" in result.text
    assert result.note


def test_filename_path_is_stripped():
    assert att.process_file("../../etc/x.txt", b"abc").filename == "x.txt"


def test_process_base64_accepts_data_url():
    raw = base64.b64encode(b"tekst").decode()
    assert att.process_base64("a.txt", raw).text == "tekst"
    assert att.process_base64("a.txt", "data:text/plain;base64," + raw).text == "tekst"


def test_validate_image_data_url():
    good = att.process_file("x.png", make_image("PNG")).images[0]
    assert att.validate_image_data_url(good) == good
    png_url = "data:image/png;base64," + base64.b64encode(make_image("PNG")).decode()
    assert att.validate_image_data_url(png_url).startswith("data:image/jpeg")
    with pytest.raises(att.AttachmentError):
        att.validate_image_data_url("https://example.com/a.png")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def test_ensure_vision_model():
    assert att.ensure_vision_model({"model": "deepseek/deepseek-chat"}, False) == {
        "model": "deepseek/deepseek-chat"
    }
    switched = att.ensure_vision_model(
        {"model": "deepseek/deepseek-chat", "temperature": 0.2}, True
    )
    assert switched["model"] == att.VISION_FALLBACK_MODEL
    assert switched["temperature"] == 0.2
    kept = att.ensure_vision_model({"model": "google/gemini-2.5-flash"}, True)
    assert kept["model"] == "google/gemini-2.5-flash"
    assert att.ensure_vision_model(None, True)["model"] == att.VISION_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_build_messages_puts_attachment_text_in_user_message():
    messages = build_messages(
        "Ile ECTS?",
        ["kontekst z bazy"],
        attachments=[
            {
                "filename": "plan.pdf",
                "kind": "pdf",
                "text": "Analiza 5 ECTS",
                "images": [],
            }
        ],
    )
    last = messages[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], str)
    assert '<<<ZAŁĄCZNIK: "plan.pdf" (pdf)>>>' in last["content"]
    assert "Analiza 5 ECTS" in last["content"]
    assert last["content"].rstrip().endswith("Ile ECTS?")
    assert "Załączniki" in messages[0]["content"]  # system rule present


def test_build_messages_sends_images_as_image_parts():
    url = "data:image/jpeg;base64,AAAA"
    messages = build_messages(
        "Co jest na zdjęciu?",
        [],
        attachments=[
            {"filename": "a.jpg", "kind": "image", "text": None, "images": [url]}
        ],
    )
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1] == {"type": "image_url", "image_url": {"url": url}}


def test_history_replays_attachment_text_but_not_images():
    history = [
        Message(
            role="user",
            content="Streść plik",
            attachments=[
                Attachment(filename="notatki.txt", kind="text", text="Ważna treść"),
                Attachment(
                    filename="foto.jpg",
                    kind="image",
                    images=["data:image/jpeg;base64,AAAA"],
                ),
            ],
        ),
        Message(role="assistant", content="Streszczenie"),
    ]
    messages = build_messages("A co dalej?", [], conversation_history=history)
    replayed = messages[1]
    assert set(replayed) == {"role", "content"}  # no extra keys sent to the LLM API
    assert "Ważna treść" in replayed["content"]
    assert "base64" not in replayed["content"]
    assert messages[2] == {"role": "assistant", "content": "Streszczenie"}
