"""
Endpoint tests for src/api/api.py (attachments + chat).

Retrieval, query rewriting, translation and the LLM are replaced with stubs,
so these tests need neither Qdrant nor network access.
"""

import importlib
import io
import sys
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def api(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "src.api.retrieval",
        _stub_module(
            "src.api.retrieval",
            get_top_k_chunks=lambda q: [
                {"text_chunk": "fakt z bazy", "source_url": "https://mini.pw.edu.pl"}
            ],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.api.query_rewriter",
        _stub_module("src.api.query_rewriter", rewrite_query=lambda q: q),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.api.translator",
        _stub_module(
            "src.api.translator", translate_text=lambda text, target_lang_code: text
        ),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # regex-only anonymization: no NER model download in tests
    monkeypatch.setenv("ANONYMIZER_FALLBACK", "regex")
    sys.modules.pop("src.api.main", None)
    sys.modules.pop("src.api.api", None)
    module = importlib.import_module("src.api.api")

    calls = {}

    def fake_stream(messages, model_config=None):
        calls["messages"] = messages
        calls["model_config"] = model_config
        yield "odpowiedź"

    def fake_query(messages, model_config=None):
        calls["messages"] = messages
        calls["model_config"] = model_config
        return "odpowiedź"

    monkeypatch.setattr(module, "query_llm_stream", fake_stream)
    monkeypatch.setattr(module, "query_llm", fake_query)
    monkeypatch.setattr(module, "_enrich_retrieval_query", lambda q, *a, **k: q)
    yield TestClient(module.app), calls
    sys.modules.pop("src.api.api", None)
    sys.modules.pop("src.api.main", None)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (0, 128, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_text_file(api):
    client, _ = api
    resp = client.post(
        "/attachments",
        files={"file": ("notatki.txt", b"Termin: 20 listopada", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "text"
    assert body["text"] == "Termin: 20 listopada"


def test_upload_image(api):
    client, _ = api
    resp = client.post(
        "/attachments", files={"file": ("foto.png", _png(), "image/png")}
    )
    assert resp.status_code == 200
    assert resp.json()["images"][0].startswith("data:image/jpeg;base64,")


def test_upload_unsupported_returns_polish_message(api):
    client, _ = api
    resp = client.post(
        "/attachments",
        files={"file": ("x.exe", b"MZ\x00\x00\x01", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert "Nieobsługiwany" in resp.json()["detail"]


def test_legacy_extract_text_still_works(api):
    client, _ = api
    resp = client.post(
        "/extract-text", files={"file": ("a.md", b"# Hej", "text/markdown")}
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "# Hej", "filename": "a.md"}


def test_chat_stream_with_text_attachment_keeps_retrieval_query_clean(api):
    client, calls = api
    upload = client.post(
        "/attachments", files={"file": ("plan.txt", b"Analiza: sala 101", "text/plain")}
    ).json()
    resp = client.post(
        "/chat/stream",
        json={
            "query": "Gdzie jest analiza?",
            "attachments": [upload],
            "modelConfig": {"model": "deepseek/deepseek-chat-v3-0324"},
        },
    )
    assert resp.status_code == 200
    assert "odpowiedź" in resp.text
    assert '"retrieval_query": "Gdzie jest analiza?"' in resp.text
    user_msg = calls["messages"][-1]["content"]
    assert "Analiza: sala 101" in user_msg
    # no image -> the selected model is kept
    assert calls["model_config"]["model"] == "deepseek/deepseek-chat-v3-0324"


def test_chat_stream_with_image_switches_to_vision_model(api):
    client, calls = api
    upload = client.post(
        "/attachments", files={"file": ("foto.png", _png(), "image/png")}
    ).json()
    resp = client.post(
        "/chat/stream",
        json={
            "query": "",
            "attachments": [upload],
            "modelConfig": {"model": "deepseek/deepseek-chat-v3-0324"},
        },
    )
    assert resp.status_code == 200
    assert calls["model_config"]["model"].startswith(
        ("openai/", "google/", "anthropic/")
    )
    content = calls["messages"][-1]["content"]
    assert content[1]["type"] == "image_url"


def test_chat_accepts_legacy_base64_pdf_attachment(api):
    import base64

    from src.api.tests.attachments_test import make_text_pdf

    client, calls = api
    data = base64.b64encode(make_text_pdf(["Regulamin"])).decode()
    resp = client.post(
        "/chat",
        json={
            "query": "Co jest w pliku?",
            "attachments": [
                {"filename": "r.pdf", "data": data, "mime_type": "application/pdf"}
            ],
        },
    )
    assert resp.status_code == 200
    assert "Regulamin" in calls["messages"][-1]["content"]


def test_chat_rejects_empty_query_without_attachments(api):
    client, _ = api
    assert client.post("/chat", json={"query": ""}).status_code == 400


def test_chat_rejects_too_many_attachments(api):
    client, _ = api
    many = [{"filename": f"{i}.txt", "kind": "text", "text": "x"} for i in range(6)]
    resp = client.post("/chat", json={"query": "?", "attachments": many})
    assert resp.status_code == 413


def test_attachment_text_is_masked_and_file_name_hidden(api):
    client, calls = api
    upload = client.post(
        "/attachments",
        files={
            "file": (
                "Jan_Kowalski_CV.txt",
                b"PESEL: 44051401359, e-mail jan.kowalski@example.com",
                "text/plain",
            )
        },
    ).json()
    resp = client.post(
        "/chat", json={"query": "Co jest w pliku?", "attachments": [upload]}
    )
    assert resp.status_code == 200
    prompt = calls["messages"][-1]["content"]
    assert "44051401359" not in prompt
    assert "jan.kowalski@example.com" not in prompt
    assert "[PESEL]" in prompt
    assert "Kowalski_CV" not in prompt  # real file name never reaches the model
    assert 'ZAŁĄCZNIK: "plik 1 (.txt)"' in prompt


def test_history_attachment_text_is_masked(api):
    client, calls = api
    resp = client.post(
        "/chat",
        json={
            "query": "A drugi akapit?",
            "conversation_history": [
                {
                    "role": "user",
                    "content": "Streść plik",
                    "attachments": [
                        {
                            "filename": "Nowak.txt",
                            "kind": "text",
                            "text": "PESEL 44051401359",
                        }
                    ],
                },
                {"role": "assistant", "content": "Streszczenie"},
            ],
        },
    )
    assert resp.status_code == 200
    replayed = calls["messages"][1]["content"]
    assert "44051401359" not in replayed and "[PESEL]" in replayed
    assert "Nowak" not in replayed


def test_attachment_text_counts_towards_budget(api, monkeypatch):
    client, _ = api
    module = sys.modules["src.api.api"]
    seen = {}
    original = module.guard.reserve

    def spy(client_id, endpoint, input_chars, max_tokens):
        seen["input_chars"] = input_chars
        return original(
            client_id, endpoint, input_chars=input_chars, max_tokens=max_tokens
        )

    monkeypatch.setattr(module.guard, "reserve", spy)
    big = [{"filename": "a.txt", "kind": "text", "text": "x" * 30000}]
    client.post("/chat", json={"query": "?", "attachments": big})
    assert seen["input_chars"] >= 30000
