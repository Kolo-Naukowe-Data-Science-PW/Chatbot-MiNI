"""Tests for prompt anonymization (src/api/ANONYMIZATION.md).

anonymizer.py (regex layer, filters, cache, fallback), staff_names.py (staff list)
and request_privacy.py (wiring into the endpoints). The NER model is replaced by
a fake, so no model download is needed.
"""

import base64

import pytest
from fastapi import HTTPException

from src.api.anonymizer import (
    AnonymizationUnavailable,
    Anonymizer,
    apply_masks,
    has_title,
    is_patron,
    looks_like_name,
    mask_identifiers,
    split_into_chunks,
    strip_roles,
)
from src.api.models import Message
from src.api.request_privacy import (
    UnreadablePdf,
    anonymize_request,
    anonymize_text_or_http_error,
    pdf_attachments_as_text,
    split_attachment,
)
from src.api.staff_names import (
    StaffEntry,
    StaffList,
    load_staff_list,
    name_words,
    one_typo,
    parse_staff_lines,
    parse_staff_text,
    parse_staff_urls,
)


class FakeModel:
    """Stands in for the NER model (already filtered spans): returns given fragments,
    records the texts it was asked about, can fail."""

    def __init__(self, spans_for=None, fail=False):
        self.spans_for = spans_for or {}
        self.fail = fail
        self.texts = []

    @property
    def calls(self):
        return len(self.texts)

    def __call__(self, text):
        self.texts.append(text)
        if self.fail:
            raise TimeoutError("model down")
        return [
            s for key, spans in self.spans_for.items() if key in text for s in spans
        ]


def make(monkeypatch, fallback="block", **fake_kwargs):
    anon = Anonymizer(fallback=fallback)
    fake = FakeModel(**fake_kwargs)
    monkeypatch.setattr(anon, "_model_spans", fake)
    return anon, fake


def make_raw(monkeypatch, entities):
    """Fake returning raw model entities, so the filters in the pipeline are tested too."""
    anon = Anonymizer(fallback="block")
    monkeypatch.setattr(
        anon, "_detect", lambda text: [e for e in entities if e.lower() in text.lower()]
    )
    return anon


# ==========================================================================
# regex layer: identifiers
# ==========================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        # identifiers - masked on FORMAT, also fake / typo values
        ("Mój PESEL to 44051401458, co dalej?", "Mój PESEL to [PESEL], co dalej?"),
        ("PESEL 12345678901 (zmyślony)", "PESEL [PESEL] (zmyślony)"),
        ("pisz na jan.kowalski@pw.edu.pl", "pisz na [EMAIL]"),
        ("tel. +48 600 123 456", "tel. [TELEFON]"),
        ("dzwoń 600-123-456", "dzwoń [TELEFON]"),
        ("NIP 526-104-08-28", "NIP [NIP]"),
        ("dowód ABC123456", "dowód [NR_DOWODU]"),
        ("konto PL61 1090 1014 0000 0712 1981 2874", "konto [NR_KONTA]"),
        ("karta 4111 1111 1111 1111", "karta [NR_KARTY]"),
        ("mam nr albumu 312345 i pytanie", "mam nr albumu [NR_ALBUMU] i pytanie"),
        ("paszport nr AB1234567", "paszport nr [NR_PASZPORTU]"),
        ("prawo jazdy 00123/12/1234", "prawo jazdy [NR_PRAWA_JAZDY]"),
        ("moje IP 192.168.1.10", "moje IP [IP]"),
        ("hasło do USOS to Kot123!", "hasło do USOS to [HASLO]"),
        ("urodziłem się 12.03.2004 r.", "urodziłem się [DATA_URODZENIA] r."),
        ("data urodzenia: 3 maja 2003", "data urodzenia: [DATA_URODZENIA]"),
        # addresses
        (
            "adres zamieszkania: Mokotowska 12/4, Warszawa",
            "adres zamieszkania: [ADRES], Warszawa",
        ),
        ("mieszkam na Mokotowskiej 12/4", "mieszkam na [ADRES]"),
        ("mój adres to Złota 44", "mój adres to [ADRES]"),
        # catch-all
        ("numer sprawy 98765432", "numer sprawy [NUMER]"),
        # spaced / dashed formats
        (
            "Mój pesel: 0227 0812 345, co z wnioskiem?",
            "Mój pesel: [PESEL], co z wnioskiem?",
        ),
        ("pesel 022708-12345 czy dobrze?", "pesel [PESEL] czy dobrze?"),
        ("PESEL to 440 514 014 58", "PESEL to [PESEL]"),
        (
            "Numer indeksu: 31 23 45, czy mam ocenę?",
            "Numer indeksu: [NR_ALBUMU], czy mam ocenę?",
        ),
        ("nr albumu 312-345", "nr albumu [NR_ALBUMU]"),
        (
            "Mieszkam w Pruszkowie przy Długiej 5, czy dostanę akademik?",
            "Mieszkam w Pruszkowie przy [ADRES], czy dostanę akademik?",
        ),
        ("mieszkam w akademiku na Kickiego 12", "mieszkam w akademiku na [ADRES]"),
        # addresses with flat number, postal code, dorm room
        ("Mój adres to Krucza 14 m. 8, 00-500 Warszawa", "Mój adres to [ADRES]"),
        (
            "Adres zamieszkania: ul. Polna 3, 05-800 Pruszków, co dalej?",
            "Adres zamieszkania: [ADRES], co dalej?",
        ),
        (
            "Wyślijcie dokumenty na adres ul. Sienna 10/2, 00-121 Warszawa",
            "Wyślijcie dokumenty na adres [ADRES]",
        ),
        (
            "Jestem zameldowany na Grochowskiej 220, czy to ważne?",
            "Jestem zameldowany na [ADRES], czy to ważne?",
        ),
        ("Mieszkam w akademiku Riviera, pokój 312", "Mieszkam w [ADRES]"),
        (
            "Mieszkam w akademiku Riviera, czy mogę zmienić pokój?",
            "Mieszkam w [ADRES], czy mogę zmienić pokój?",
        ),
        ("mój adres: Lipowa 7 lok. 3", "mój adres: [ADRES]"),
    ],
)
def test_detects(text, expected):
    assert mask_identifiers(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "Kiedy jest egzamin z analizy w sali 101?",
        "Ile ECTS ma przedmiot 1120-IN000-ISP-0123?",
        "Rok akademicki 2025/2026, semestr 3",
        "Gdzie jest wydział? ul. Koszykowa 75, 00-662 Warszawa",
        "Co się znajduje w Rektorskiej 4?",
        "Jak dojść na ul. Rektorską 4 z metra?",
        "Kto prowadzi wykład, dr Kowalski?",  # names left alone by design
        "Zajęcia w poniedziałek 10:15-12:00, termin 12.03.2027",  # date w/o birth context
        "Mam depresję, czy mogę wziąć urlop dziekański?",  # content kept, see docs
        "Ile osób ma indeks na 2 roku, 120 czy 150?",  # numbers after "indeks", not an ID
        "Mieszkam w Pruszkowie, jak dojechać na wydział?",  # town only, no street number
        "Czy mieszkam w strefie 2, jeśli dojeżdżam z Piaseczna?",  # no "na/przy" + address
        "Czy mogę mieszkać w akademiku Riviera?",  # not first person present
        "Jaki jest adres dziekanatu?",  # no number
        "Na adres mailowy przyszło potwierdzenie z numerem 2",  # "na adres" only before ul./al./pl.
    ],
)
def test_no_false_positives(text):
    assert mask_identifiers(text).text == text


def test_counts_only_labels():
    r = mask_identifiers("44051401458 i 44051401458")
    assert r.found == {"PESEL": 2}
    assert "44051401458" not in str(r.found)


# English / Ukrainian (the chatbot has PL / EN / UA; all rules run on every text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My phone is +380 67 123 4567", "My phone is [TELEFON]"),
        ("Call 0044 20 7946 0958", "Call [TELEFON]"),
        ("My student ID is 312345", "My student ID is [NR_ALBUMU]"),
        ("номер студентського квитка 312345", "номер студентського квитка [NR_ALBUMU]"),
        ("my password is Kot123!", "my password is [HASLO]"),
        ("Мій пароль: Kot123!", "Мій пароль: [HASLO]"),
        ("I was born on March 3, 2004", "I was born on [DATA_URODZENIA]"),
        ("Я народився 3 березня 2004", "Я народився [DATA_URODZENIA]"),
        (
            "I live at Mokotowska 12/4, can I get a dorm?",
            "I live at [ADRES], can I get a dorm?",
        ),
        (
            "I live at 12 Mokotowska St, can I get a dorm?",
            "I live at [ADRES], can I get a dorm?",
        ),
        ("Я живу на вулиці Мокотовській 12/4, чи можу?", "Я живу на [ADRES], чи можу?"),
        ("паспорт КВ123456", "паспорт [NR_PASZPORTU]"),
    ],
)
def test_detects_en_uk(text, expected):
    assert mask_identifiers(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "I live in a dorm, is room 205 free?",
        "Exam on March 3, 2027 in room 101",
        "What is the password policy in USOS?",
        "Is the student ID card valid abroad?",
        "+5 ECTS za projekt",
    ],
)
def test_no_false_positives_en_uk(text):
    assert mask_identifiers(text).text == text


# ==========================================================================
# apply_masks
# ==========================================================================


def test_whole_word_is_masked_not_a_fragment():
    # model returned the base form, text has the inflected one
    out = apply_masks("Jakie są konsultacje dr. hab. Wiśniewskiego?", ["Wiśniewski"])
    assert out == "Jakie są konsultacje dr. hab. [OSOBA]?"


def test_existing_placeholder_is_never_touched():
    out = apply_masks(
        "Jestem Julia Nowicka, nr albumu [NR_ALBUMU].", ["NR_ALBUMU", "Julia Nowicka"]
    )
    assert out == "Jestem [OSOBA], nr albumu [NR_ALBUMU]."


def test_overlapping_spans_are_merged():
    out = apply_masks("Pozdrawiam, Tomasz Wójcik", ["Tomasz Wójcik", "Wójcik"])
    assert out == "Pozdrawiam, [OSOBA]"


def test_case_insensitive_and_all_occurrences():
    out = apply_masks("jestem michał mazur. Michał Mazur pyta.", ["Michał Mazur"])
    assert out == "jestem [OSOBA]. [OSOBA] pyta."


def test_too_short_or_bracketed_spans_are_ignored():
    text = "Sala 101 [PESEL]"
    assert apply_masks(text, ["1", "[PESEL]"]) == text


# ==========================================================================
# who is not masked: titles, roles, patrons
# ==========================================================================


@pytest.mark.parametrize(
    "span, text",
    [
        ("prof. Tomaszewski", "Kiedy prof. Tomaszewski ma dyżur?"),  # title inside span
        ("Tomaszewski", "Kiedy prof. Tomaszewski ma dyżur?"),  # title right before
        ("Anna Kwiatkowska", "Czy dr hab. Anna Kwiatkowska prowadzi seminarium?"),
        ("Walczakiem", "Jak skontaktować się z dr Walczakiem?"),
        ("Brzeziński", "Jaki temat proponuje dr hab. inż. Brzeziński?"),
        ("Czerwińskiego", "Konsultacje profesora Czerwińskiego?"),
        (
            "Jan Sobolewski",
            "Mój promotor dr Jan Sobolewski nie odpisuje, co robić?",
        ),  # "mój" not right before title
    ],
)
def test_titled_names_are_staff(span, text):
    assert has_title(span, text)


@pytest.mark.parametrize(
    "span, text",
    [
        (
            "Oskar Wojciechowski",
            "Mój kolega Oskar Wojciechowski pyta, czy prof. Tomaszewski przyjmuje projekty.",
        ),
        ("Anna Zielińska", "Nazywam się Anna Zielińska."),
        ("Drugi", "Drugi termin egzaminu?"),  # "dr" must be a whole word
        # "pan/pani" does not protect: also used for private people
        ("Ewa Kurek", "Moja mama, pani Ewa Kurek, pyta o rozdanie dyplomów."),
        (
            "Marzena",
            "Czy pani Marzena z dziekanatu pracuje w piątki?",
        ),  # left to the model
        # a title after "jestem / nazywam się / mój / moja..." belongs to the author
        ("Piotr Zieliński", "Jestem inż. Piotr Zieliński, student II stopnia."),
        ("inż. Piotr Zieliński", "Jestem inż. Piotr Zieliński, student II stopnia."),
        ("Anna Pietrasik", "Nazywam się mgr Anna Pietrasik, jestem doktorantką."),
        ("Ewa Kurek", "Pozdrawiam, mgr inż. Ewa Kurek"),
    ],
)
def test_untitled_names_are_not_staff(span, text):
    assert not has_title(span, text)


@pytest.mark.parametrize(
    "span, expected",
    [
        ("Anna Nowak", True),
        ("kasia nowicka", True),  # lowercase private name stays
        ("Олена Коваль", True),
        ("student Wydziału MiNI", False),  # starts with a role word
        ("Pełnomocnikowi ds. Praktyk", False),
        ("nauczyciela akademickiego", False),
        ("dodatkowego recenzenta", False),  # role word not in the first position
        ("uprawnione osoby", False),
        ("dany wykładowca/ćwiczeniowiec", False),
        ("polyglot students", False),
        ("grupy 1", False),  # digits
        ("Kto jest koordynatorem przedmiotu Reprezentacja wiedzy", False),  # too long
    ],
)
def test_looks_like_name(span, expected):
    assert looks_like_name(span) is expected


@pytest.mark.parametrize(
    "span, expected",
    [
        ("Pani Anna Zielińska", "Anna Zielińska"),
        ("brat Piotr Zieliński", "Piotr Zieliński"),
        ("Dziekan Anna Zielińska", "Anna Zielińska"),
        ("rektor Przykładowy", "Przykładowy"),
        ("Jan Panek", "Jan Panek"),  # surnames that start like a role word stay
        ("Bratkowski", "Bratkowski"),
        ("Theodor Myśliwiec", "Theodor Myśliwiec"),
        ("Pani", ""),
        (
            "student Wydziału MiNI",
            "Wydziału MiNI",
        ),  # the pipeline then asks the model about the rest alone
    ],
)
def test_strip_roles(span, expected):
    assert strip_roles(span) == expected


@pytest.mark.parametrize(
    "span, text",
    [
        ("Mikołaja Kopernika", "Gdzie jest sala im. Mikołaja Kopernika?"),
        ("Banacha", "Czy twierdzenie Banacha będzie na egzaminie?"),
        ("Newtona", "Czy omawiamy metodę Newtona?"),
        (
            "Stefana Banacha",
            "Czy stypendium im. Stefana Banacha mogą dostać studenci MiNI?",
        ),
        ("Noakowskiego", "Czy biblioteka przy ul. Noakowskiego jest czynna?"),
    ],
)
def test_patrons_are_kept(span, text):
    assert is_patron(span, text)


def test_private_person_is_not_a_patron():
    assert not is_patron("Anna Nowak", "Piszę w imieniu Anny Nowak, która jest chora.")
    assert not is_patron(
        "Anny Nowak", "W sali 101 Anny Nowak nie było."
    )  # "sali 101" is not right before


# ==========================================================================
# Anonymizer: regex + model + cache + fallback
# ==========================================================================


def test_regex_then_model(monkeypatch):
    anon, _ = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    r = anon.anonymize("Jestem Anna Nowak, PESEL 44051401458.")
    assert r.text == "Jestem [OSOBA], PESEL [PESEL]."
    assert r.found == {"PESEL": 1, "OSOBA": 1}
    assert not r.regex_only


def test_cache_calls_model_once_per_text(monkeypatch):
    anon, fake = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    for _ in range(3):
        assert anon.anonymize("Jestem Anna Nowak").text == "Jestem [OSOBA]"
    assert fake.calls == 1
    assert anon.stats["cache_hit"] == 2


def test_cache_is_bounded_and_keyed_by_hash(monkeypatch):
    anon, _ = make(monkeypatch)
    anon.cache_size = 2
    for t in ("a1", "a2", "a3"):
        anon.anonymize(t)
    assert len(anon._cache) == 2
    assert all(len(k) == 64 for k in anon._cache)  # sha256 hex keys, not the raw text
    assert not any(k in ("a1", "a2", "a3") for k in anon._cache)


def test_history_only_user_messages(monkeypatch):
    anon, fake = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    history = [
        {"role": "user", "content": "Jestem Anna Nowak, kiedy sesja?"},
        {"role": "assistant", "content": "Sesja zaczyna się w lutym, Anno Nowak."},
    ]
    out = anon.anonymize_history(history)
    assert out[0]["content"] == "Jestem [OSOBA], kiedy sesja?"
    assert out[1] == history[1]  # bot answers untouched
    assert fake.calls == 1
    anon.anonymize_history(
        history
    )  # frontend re-sends the history with the next question
    assert fake.calls == 1  # served from cache


def test_history_with_pydantic_like_objects(monkeypatch):
    class Msg:
        def __init__(self, role, content):
            self.role, self.content = role, content

        def model_copy(self, update):
            return Msg(self.role, update.get("content", self.content))

    anon, _ = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    out = anon.anonymize_history([Msg("user", "Jestem Anna Nowak")])
    assert out[0].content == "Jestem [OSOBA]"


def test_fallback_block_raises(monkeypatch):
    anon, _ = make(monkeypatch, fallback="block", fail=True)
    with pytest.raises(AnonymizationUnavailable):
        anon.anonymize("Jestem Anna Nowak, PESEL 44051401458.")
    assert anon.stats["blocked"] == 1


def test_fallback_regex_masks_identifiers_only(monkeypatch):
    anon, _ = make(monkeypatch, fallback="regex", fail=True)
    r = anon.anonymize("Jestem Anna Nowak, PESEL 44051401458.")
    assert r.text == "Jestem Anna Nowak, PESEL [PESEL]."
    assert r.regex_only
    assert anon.stats["fallback_regex"] == 1
    assert not anon._cache  # a fallback result is never cached


def test_logs_never_contain_the_prompt(monkeypatch, caplog):
    anon, _ = make(monkeypatch, fallback="regex", fail=True)
    with caplog.at_level("DEBUG"):
        anon.anonymize("Jestem Anna Nowak, PESEL 44051401458.")
    assert caplog.records  # the fallback is logged...
    assert (
        "Anna" not in caplog.text and "44051401458" not in caplog.text
    )  # ...not the prompt


def test_invalid_fallback_setting():
    with pytest.raises(ValueError):
        Anonymizer(fallback="send-anyway")


def test_warm_up_never_raises(monkeypatch):
    anon, _ = make(monkeypatch, fail=True)
    assert anon.warm_up() is False  # failed, but did not raise
    anon2, _ = make(monkeypatch)
    assert anon2.warm_up() is True


def test_failed_model_load_is_retried_after_a_pause(monkeypatch):
    anon = Anonymizer(fallback="block")
    attempts = []
    monkeypatch.setattr(
        anon,
        "_load_model",
        lambda: attempts.append(1) or (_ for _ in ()).throw(OSError("no net")),
    )
    with pytest.raises(AnonymizationUnavailable):
        anon.anonymize("Jestem Anna Nowak")
    with pytest.raises(AnonymizationUnavailable):
        anon.anonymize("Jestem Anna Kowal")
    assert len(attempts) == 1 and anon.stats["load_failed"] == 1  # no retry storm
    monkeypatch.setattr(anon, "RETRY_SECONDS", 0)
    with pytest.raises(AnonymizationUnavailable):
        anon.anonymize("Jestem Anna Lis")
    assert len(attempts) == 2


def test_empty_text_skips_model(monkeypatch):
    anon, fake = make(monkeypatch)
    assert anon.anonymize("   ").text == "   "
    assert fake.calls == 0


# filters on raw model output


def test_rest_after_a_role_word_must_be_a_person_on_its_own(monkeypatch):
    # the fake model returns the listed entities that occur in the text it is asked about
    anon = make_raw(
        monkeypatch,
        [
            "pani anna zielińska",
            "anna zielińska",
            "student Wydziału MiNI",
            "osobę w rodzinie",
        ],
    )
    assert (
        anon.anonymize("pani anna zielińska z mojego roku zgubiła legitymację").text
        == "pani [OSOBA] z mojego roku zgubiła legitymację"
    )
    assert (
        anon.anonymize(
            "Co robi student Wydziału MiNI, dochód na osobę w rodzinie?"
        ).text
        == "Co robi student Wydziału MiNI, dochód na osobę w rodzinie?"
    )


def test_role_word_in_the_span_does_not_hide_a_private_person(monkeypatch):
    anon = make_raw(
        monkeypatch,
        [
            "Pani Anna Zielińska",
            "Anna Zielińska",
            "brat Piotr Zieliński",
            "Piotr Zieliński",
            "doktorant Paweł Zając",
            "Paweł Zając",
        ],
    )
    out = anon.anonymize(
        "Pani Anna Zielińska i mój brat Piotr Zieliński pytają, a doktorant Paweł Zając?"
    ).text
    assert out == "Pani [OSOBA] i mój brat [OSOBA] pytają, a doktorant [OSOBA]?"


def test_add_staff_drops_cached_results(monkeypatch):
    anon = make_raw(monkeypatch, ["Adam Przykładowy"])
    assert anon.anonymize("Kim jest Adam Przykładowy?").text == "Kim jest [OSOBA]?"
    # names from the knowledge base arrive later
    anon.add_staff([StaffEntry("Przykładowy", "Adam")])
    assert (
        anon.anonymize("Kim jest Adam Przykładowy?").text
        == "Kim jest Adam Przykładowy?"
    )


def test_filters_in_the_pipeline(monkeypatch):
    anon = make_raw(
        monkeypatch,
        [
            "Anna Nowak",
            "Banacha",
            "Kaczmarskiego",
            "dr. Walczakiem",
            "student Wydziału MiNI",
        ],
    )
    out = anon.anonymize(
        "Jestem Anna Nowak, student Wydziału MiNI. Czy twierdzenie Banacha będzie u Kaczmarskiego "
        "albo u dr. Walczakiem?"
    ).text
    assert out == (
        "Jestem [OSOBA], student Wydziału MiNI. Czy twierdzenie Banacha będzie u Kaczmarskiego "
        "albo u dr. Walczakiem?"
    )


def test_staff_list_keeps_staff_and_masks_students(monkeypatch):
    # Kaczmarski / Keller / Mańdziuk have staff pages in links_extended.py; Kellner and Mazur do not
    anon = make_raw(
        monkeypatch,
        [
            "Krzysztofem Kaczmarskim",
            "Kellerowi",
            "Jacek Mandziuk",
            "Kellner",
            "Tomasz Mazur",
        ],
    )
    out = anon.anonymize(
        "Mam zajęcia z Krzysztofem Kaczmarskim, oddałem Kellerowi, Jacek Mandziuk też. "
        "Kellner i Tomasz Mazur z mojej grupy pytają."
    ).text
    assert out == (
        "Mam zajęcia z Krzysztofem Kaczmarskim, oddałem Kellerowi, Jacek Mandziuk też. "
        "[OSOBA] i [OSOBA] z mojej grupy pytają."
    )


def test_hallucinated_span_is_ignored(monkeypatch):
    anon = make_raw(monkeypatch, ["Anna Nowak"])
    monkeypatch.setattr(
        anon, "_detect", lambda text: ["Zofia Wrona"]
    )  # not in the text
    assert anon.anonymize("Kiedy jest sesja?").text == "Kiedy jest sesja?"


# long texts: chunking


def test_short_text_is_one_chunk():
    assert split_into_chunks("Kiedy jest sesja?", 1500) == ["Kiedy jest sesja?"]


def test_chunks_respect_limit_and_keep_all_text():
    text = "\n".join(
        f"Zdanie numer {i} o zajęciach u dr Jana Kowalskiego." for i in range(80)
    )
    chunks = split_into_chunks(text, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert all("dr Jana Kowalskiego" in c for c in chunks)  # title and name together


def test_very_long_sentence_is_cut_at_spaces():
    chunks = split_into_chunks("słowo " * 1000, 500)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 1


def test_long_text_calls_model_per_chunk(monkeypatch):
    anon, fake = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    anon.chunk_chars, anon.max_chunks = 200, 10
    text = ("Pytanie o plan zajęć na trzecim semestrze. " * 10) + "Jestem Anna Nowak."
    r = anon.anonymize(text)
    assert fake.calls == len(split_into_chunks(text, 200)) > 1
    assert "Anna Nowak" not in r.text and "[OSOBA]" in r.text


def test_too_long_text_follows_fallback(monkeypatch):
    anon, fake = make(monkeypatch, fallback="block")
    anon.chunk_chars, anon.max_chunks = 100, 2
    with pytest.raises(AnonymizationUnavailable):
        anon.anonymize("Mój PESEL 44051401458. " + "bla " * 200)
    assert fake.calls == 0 and anon.stats["too_long"] == 1

    anon, fake = make(monkeypatch, fallback="regex")
    anon.chunk_chars, anon.max_chunks = 100, 2
    r = anon.anonymize("Mój PESEL 44051401458. " + "bla " * 200)
    assert r.regex_only and "[PESEL]" in r.text and fake.calls == 0


# ==========================================================================
# staff list (staff_names.py)
# ==========================================================================


def test_parse_staff_urls():
    urls = [
        "https://ww2.mini.pw.edu.pl/wydzial/pracownicy/prac_balicki_jerzy/",
        "https://ww2.mini.pw.edu.pl/wydzial/pracownicy/prac_dabrowski-tumanski_pawel/",
        "https://ww2.mini.pw.edu.pl/wydzial/pracownicy/prac_balicki_jerzy/",  # duplicate
        "https://ww2.mini.pw.edu.pl/studia/dziekanat/kontakt-i-dyzury/",  # not a staff page
    ]
    assert parse_staff_urls(urls) == [
        StaffEntry("balicki", "jerzy"),
        StaffEntry("dabrowski-tumanski", "pawel"),
    ]


def test_parse_staff_text_from_knowledge_base():
    facts = [
        "Dziekan: prof. dr hab. Grzegorz Świątek. Prodziekan ds. Studenckich: dr hab. inż. Agata Pilitowska, prof. uczelni.",
        "Konsultacje dr. Jana Kowalskiego odbywają się we wtorki.",  # genitive, stored as found
        "Kierownik Działu Promocji: Anna Maria Nowak, pokój 123.",  # function + colon, three-word name
        "Rektor PW prof. dr hab. inż. Adam Przykładowy powołał komisję.",
        "Dziekan Wydziału Matematyki ogłosił terminy sesji.",  # no separator: no name captured
        "Zajęcia prowadzi mgr Ewa Lis-Kowalska.",
    ]
    entries = parse_staff_text(facts)
    assert StaffEntry("Świątek", "Grzegorz") in entries
    assert StaffEntry("Pilitowska", "Agata") in entries
    assert StaffEntry("Kowalskiego", "Jana") in entries
    assert StaffEntry("Nowak", "Anna") in entries
    assert StaffEntry("Nowak", "Maria") in entries
    assert StaffEntry("Przykładowy", "Adam") in entries
    assert StaffEntry("Lis-Kowalska", "Ewa") in entries
    assert not any(e.surname == "Matematyki" for e in entries)
    staff = StaffList(entries)
    assert staff.is_staff("Przykładowy") and staff.is_staff("Adam Przykładowy")
    assert staff.is_staff("Kowalski")
    assert staff.is_staff("Anna Nowak") and staff.is_staff("Marii Nowak")
    assert not staff.is_staff("Jan Nowak")  # other first name


def test_staff_list_add_skips_duplicates():
    staff = StaffList([StaffEntry("kaczmarski", "krzysztof")])
    staff.add([StaffEntry("kaczmarski", "krzysztof"), StaffEntry("Nowak", "Jan")])
    assert len(staff) == 2 and staff.is_staff("Janowi Nowakowi")


def test_parse_staff_lines():
    lines = ["# komentarz", "", "Anna Kowalczyk", "Nowak", "  mgr Ewa Lis  "]
    assert parse_staff_lines(lines) == [
        StaffEntry("Kowalczyk", "Anna"),
        StaffEntry("Nowak"),
        StaffEntry("Lis", "Ewa"),
    ]


def test_name_words_drop_titles_and_prepositions():
    assert name_words("dr hab. z Bogucką") == ["Bogucką"]
    assert name_words("Krzysztofem Kaczmarskim") == ["Krzysztofem", "Kaczmarskim"]


STAFF = StaffList(
    [
        StaffEntry("kaczmarski", "krzysztof"),
        StaffEntry("keller", "pawel"),
        StaffEntry("mandziuk", "jacek"),
        StaffEntry("bogucka", "magdalena"),
        StaffEntry("kozlowski", "marek"),
        StaffEntry("burchart", "dagmara"),
        StaffEntry("jasionowska-skop", "magdalena"),
        StaffEntry("tur", "szymon"),
        StaffEntry("Nowak"),
    ]
)


@pytest.mark.parametrize(
    "span",
    [
        # declension
        "Kaczmarski",
        "Kaczmarskiego",
        "Kaczmarskiemu",
        "Kaczmarskim",
        "Krzysztof Kaczmarski",
        "Krzysztofa Kaczmarskiego",
        "Krzysztofem Kaczmarskim",
        # masculine noun, no diacritics
        "Kellerowi",
        "Kellera",
        "Pawła Kellera",
        "Pawla Kellera",
        # feminine adjective
        "Bogucką",
        "Boguckiej",
        "Magdalenie Boguckiej",
        # feminine surname does not decline
        "Dagmarą Burchart",
        "Burchart",
        "Mańdziuk",
        "Mandziuka",
        "Jackowi Mańdziukowi",
        "Jacek Mandziuk",
        "Jasionowskiej-Skop",
        "Magdaleną Jasionowską-Skop",
        "Tura",
        "Szymonowi Turowi",
        # one typo + matching first name
        "Krzysztof Kaczmarsiego",
        "Pawel Kelelr",
        "Krzysztofowi Kaczmarskieu",
        "Krzystof Kaczmarski",  # typo in the first name
        # list entry without a first name: any first name
        "Nowak",
        "Anna Nowak",
    ],
)
def test_is_staff(span):
    assert STAFF.is_staff(span)


@pytest.mark.parametrize(
    "span",
    [
        # near misses of bare surnames stay private
        "Kellner",
        "Kaczmarsko",
        "Kaczmarek",
        "Kozłowska",
        # other first name
        "Paweł Kozłowski",
        "Jan Keller",
        "Anna Kaczmarska",
        "Tomasz Mazur",
        "Anna Kowalczyk",
        "Олена Коваль",
        "Turr",  # bare surname, one typo: not accepted
        "",
        "z",
        "dr",
    ],
)
def test_is_not_staff(span):
    assert not STAFF.is_staff(span)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("lewandowksi", "lewandowski", True),
        ("lewandowsski", "lewandowski", True),
        # a different / missing LAST letter is another name, not a typo
        ("lewandowsk", "lewandowski", False),
        ("kozłowski", "kozłowska", False),
        ("rybakk", "rybak", True),
        ("rybaka", "rybak", False),
        ("kaczmarski", "kaczmarski", False),
    ],
)
def test_one_typo(a, b, expected):
    assert one_typo(a, b) is expected


def test_load_staff_list_from_repo_links(tmp_path):
    extra = tmp_path / "staff.txt"
    extra.write_text("# dziekanat\nAnna Testowa-Dziekanat\n", encoding="utf-8")
    staff = load_staff_list(extra)
    assert len(staff) > 150  # staff pages in links_extended.py
    assert staff.is_staff("Chełmińskiego") and staff.is_staff("Jerzego Balickiego")
    assert staff.is_staff("Anny Testowej-Dziekanat")
    assert not staff.is_staff("Jan Kowalczyk-Testowy")


def test_load_staff_list_never_raises_on_missing_file():
    assert len(load_staff_list("/nie/ma/takiego/pliku.txt")) > 150


# ==========================================================================
# wiring into the endpoints (request_privacy.py)
# ==========================================================================


class Att:
    def __init__(self, data, filename="Jan_Kowalski.pdf", mime_type="application/pdf"):
        self.data, self.filename, self.mime_type = data, filename, mime_type


class Req:
    def __init__(self, query, history=(), attachments=(), language="pl"):
        self.query = query
        self.conversation_history = list(history)
        self.attachments = list(attachments)
        self.language = language


def pdf_b64(text: str) -> str:
    """A minimal one-page PDF with a real text layer, built by hand (ASCII text only).

    Written byte by byte on purpose: no dependency on pypdf's writer API, so the
    test does not depend on the pypdf version, only reading (as in production) is used.
    """
    content = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return base64.b64encode(bytes(out)).decode()


def test_split_pasted_text_file():
    q, a = split_attachment(
        "Co z tym?\n\n[Załącznik: Jan_Kowalski.txt]\nPESEL 02270812345"
    )
    assert q == "Co z tym?" and a == "PESEL 02270812345"


def test_split_pdf_without_question():
    assert split_attachment("[Załącznik: Jan_Kowalski.pdf]") == ("", "")


def test_split_text_file_without_question():
    assert split_attachment("\n\n[Załącznik: notatki.md]\ntreść") == ("", "treść")


def test_no_marker():
    assert split_attachment("kiedy jest sesja?") == ("kiedy jest sesja?", None)


@pytest.mark.parametrize("word", ["Załącznik", "Attachment", "Вкладення"])
def test_marker_in_every_interface_language(word):
    # the frontend writes the marker in the interface language (i18n.js)
    assert split_attachment(f"What is this?\n\n[{word}: Jan_Kowalski.txt]\nPESEL") == (
        "What is this?",
        "PESEL",
    )
    assert split_attachment(f"[{word}: Jan_Kowalski.pdf]") == ("", "")


def test_question_goes_to_model_attachment_to_regex_and_file_name_is_dropped(
    monkeypatch,
):
    anon, fake = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    query, _, found = anonymize_request(
        anon,
        Req(
            "Jestem Anna Nowak, co z tym?\n\n[Załącznik: Anna_Nowak_cv.txt]\nmój PESEL 02270812345"
        ),
    )
    assert "Anna" not in query and "02270812345" not in query and "cv.txt" not in query
    assert query == "Jestem [OSOBA], co z tym?\n\n[Załącznik]\nmój PESEL [PESEL]"
    # the attachment never reaches the model
    assert fake.texts == ["Jestem Anna Nowak, co z tym?"]
    assert found == {"OSOBA": 1, "PESEL": 1}


def test_long_attachment_does_not_block(monkeypatch):
    anon, fake = make(monkeypatch)
    long_file = "Linia tekstu bez danych osobowych.\n" * 2000  # above the chunk limit
    query, _, _ = anonymize_request(
        anon, Req("Streść\n\n[Załącznik: a.txt]\n" + long_file)
    )
    assert query.startswith("Streść\n\n[Załącznik]\n") and len(fake.texts) == 1


def test_history_user_messages_masked_with_attachments(monkeypatch):
    anon, _ = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    long_file = "x " * 5000 + "tel. 600 700 800"
    history = [
        Message(
            role="user", content="Jestem Anna Nowak\n\n[Załącznik: a.txt]\n" + long_file
        ),
        Message(role="assistant", content="Odpowiedź o Anna Nowak zostaje"),
    ]
    _, hist, _ = anonymize_request(anon, Req("a dalej?", history=history))
    assert hist[0].content.startswith("Jestem [OSOBA]\n\n[Załącznik]\n")
    assert "600 700 800" not in hist[0].content
    assert hist[0].content.endswith("[TELEFON]")
    assert hist[1].content == "Odpowiedź o Anna Nowak zostaje"  # bot answers untouched


def test_history_uses_cache(monkeypatch):
    anon, fake = make(monkeypatch)
    anonymize_request(anon, Req("pytanie 1"))
    anonymize_request(
        anon, Req("pytanie 2", history=[Message(role="user", content="pytanie 1")])
    )
    assert fake.texts == ["pytanie 1", "pytanie 2"]


@pytest.mark.parametrize(
    "lang,word", [("pl", "Filtr"), ("en", "privacy"), ("ua", "Фільтр")]
)
def test_model_down_gives_503_in_user_language(monkeypatch, lang, word):
    anon, _ = make(monkeypatch, fail=True)
    with pytest.raises(HTTPException) as exc:
        anonymize_request(anon, Req("Jestem Anna Nowak", language=lang))
    assert exc.value.status_code == 503 and word in exc.value.detail


def test_history_model_down_gives_503(monkeypatch):
    anon, fake = make(monkeypatch)
    anon.anonymize("pytanie")  # cached
    fake.fail = True
    with pytest.raises(HTTPException) as exc:
        anonymize_request(
            anon, Req("pytanie", history=[Message(role="user", content="nowe")])
        )
    assert exc.value.status_code == 503


def test_fallback_regex_still_answers(monkeypatch):
    anon, _ = make(monkeypatch, fallback="regex", fail=True)
    query, _, _ = anonymize_request(anon, Req("PESEL 02270812345"))
    assert query == "PESEL [PESEL]"


def test_too_long_question_gives_413(monkeypatch):
    anon, _ = make(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        anonymize_request(anon, Req("Zdanie pytania. " * 1000, language="en"))
    assert exc.value.status_code == 413 and "too long" in exc.value.detail


def test_pdf_text_is_extracted_and_masked(monkeypatch):
    anon, _ = make(monkeypatch)
    query, _, found = anonymize_request(
        anon,
        Req(
            "[Załącznik: Jan_Kowalski.pdf]",
            attachments=[Att(pdf_b64("Zaswiadczenie PESEL 02270812345"))],
        ),
    )
    assert query == "[Załącznik]\nZaswiadczenie PESEL [PESEL]"
    assert "Kowalski" not in query and found == {"PESEL": 1}


def test_pdf_with_question(monkeypatch):
    anon, _ = make(monkeypatch)
    query, _, _ = anonymize_request(
        anon, Req("Co to za dokument?", attachments=[Att(pdf_b64("Tresc"))])
    )
    assert query == "Co to za dokument?\n\n[Załącznik]\nTresc"


@pytest.mark.parametrize(
    "data",
    [pdf_b64(""), base64.b64encode(b"not a pdf").decode()],
    ids=["scan_without_text", "broken"],
)
def test_unreadable_pdf_is_rejected(data):
    with pytest.raises(UnreadablePdf):
        pdf_attachments_as_text([Att(data)])


@pytest.mark.parametrize("lang,word", [("pl", "skan"), ("en", "scan"), ("ua", "скан")])
def test_scanned_pdf_gives_422_without_model_call(monkeypatch, lang, word):
    anon, fake = make(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        anonymize_request(
            anon, Req("Co to?", attachments=[Att(pdf_b64(""))], language=lang)
        )
    assert exc.value.status_code == 422 and word in exc.value.detail.lower()
    assert fake.texts == []  # rejected before the model is called


def test_non_pdf_attachment_is_ignored():
    att = Att(pdf_b64("PESEL 02270812345"), mime_type="image/png")
    assert pdf_attachments_as_text([att])[0] == ""


def test_retrieval_text(monkeypatch):
    anon, _ = make(monkeypatch, spans_for={"Anna": ["Anna Nowak"]})
    assert (
        anonymize_text_or_http_error(anon, "Anna Nowak 600700800")
        == "[OSOBA] [TELEFON]"
    )
    anon2, _ = make(monkeypatch, fail=True)
    with pytest.raises(HTTPException) as exc:
        anonymize_text_or_http_error(anon2, "cokolwiek")
    assert exc.value.status_code == 503
