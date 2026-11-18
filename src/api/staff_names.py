"""Staff list for the anonymizer: names the chatbot knows are never masked.

Sources, all public:
1. staff pages in the scraping list (src/ingestion/links_extended.py): URLs of
   the form ``.../wydzial/pracownicy/prac_<surname>_<first-name>/``,
2. the knowledge base itself (Qdrant): every name that follows an academic
   title ("prof. dr hab. Grzegorz Świątek") or a function with a colon
   ("Kierownik dziekanatu: Anna Nowak"). If the chatbot knows a person, it can
   answer about them; if it does not, masking the name costs nothing,
3. optionally ``ANONYMIZER_STAFF_FILE``: one "Imię Nazwisko" or "Nazwisko" per
   line, "#" starts a comment.

Matching (``StaffList.is_staff``) is tolerant to Polish declension
("Kaczmarskiego" = "Kaczmarski", "z Bogucką" = "Bogucka") and to missing
diacritics ("Mandziuk" = "Mańdziuk"). A one-letter typo in the surname is
accepted only when a first name is given and it matches too ("Krzysztof
Kaczmarsiego"), never for a bare surname: a student "Kellner" must not be taken
for the lecturer "Keller". Known limit: a student with exactly the same name as
a staff member, or given by surname only, is treated as staff.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_STAFF_URL = re.compile(r"/pracownicy/prac_([a-z-]+)_([a-z-]+)/?$")

# Inflected ending -> nominative ending, applied once to the end of a word (or of
# each part of a hyphenated name). A word's "base candidates" are the word itself
# plus every rule that fits; two words match if their candidate sets intersect.
# Deliberately no plain stem/prefix match: "Kaczmarek" must not match "Kaczmarski".
_DECLENSION = [
    # adjectival surnames: Kowalskiego -> Kowalski, Jastrzębską -> Jastrzębska
    ("skiego", "ski"),
    ("skiemu", "ski"),
    ("skim", "ski"),
    ("ckiego", "cki"),
    ("ckiemu", "cki"),
    ("ckim", "cki"),
    ("dzkiego", "dzki"),
    ("dzkiemu", "dzki"),
    ("dzkim", "dzki"),
    ("skiej", "ska"),
    ("ską", "ska"),
    ("ckiej", "cka"),
    ("cką", "cka"),
    ("dzkiej", "dzka"),
    ("dzką", "dzka"),
    ("ego", "y"),
    ("emu", "y"),
    ("ym", "y"),
    ("ej", "a"),  # Jerzego -> Jerzy, Górnej -> Górna
    # feminine nouns: Dagmarą/Dagmary/Dagmarze -> Dagmara, Agnieszki/Agnieszce -> Agnieszka
    ("ą", "a"),
    ("y", "a"),
    ("i", "a"),
    ("rze", "ra"),
    ("ce", "ka"),
    ("dze", "ga"),
    ("le", "la"),
    ("nie", "na"),
    # masculine nouns: Kellera/Kellerem/Kellerowi -> Keller, Jacka -> Jacek, Pawła -> Paweł
    ("a", ""),
    ("owi", ""),
    ("em", ""),
    ("u", ""),
    ("ie", ""),
    ("ka", "ek"),
    ("kiem", "ek"),
    ("kowi", "ek"),
    ("ła", "eł"),
    ("łem", "eł"),
    ("łowi", "eł"),
    ("le", "eł"),
    ("la", "el"),
    ("lem", "el"),
    ("lowi", "el"),  # same, typed without diacritics: Pawla -> Pawel
    ("rze", "r"),
    ("cie", "t"),
]
# nominative -> inflected endings, used to compare a typo against every form of a list name
_INFLECT = {
    "ski": ("skiego", "skiemu", "skim"),
    "cki": ("ckiego", "ckiemu", "ckim"),
    "dzki": ("dzkiego", "dzkiemu", "dzkim"),
    "ska": ("skiej", "ską"),
    "cka": ("ckiej", "cką"),
    "dzka": ("dzkiej", "dzką"),
    "y": ("ego", "emu", "ym"),
}
_NAME_WORD = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+(?:-[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)*")
_TITLE_WORD = re.compile(
    r"(?i)^(?:prof|profesor\w*|dr|doktor\w*|hab|inż|mgr|magister\w*|doc|pan|pani|pana|panią|panem|panu)$"
)
_NO_DIACRITICS = str.maketrans("ąćęłńóśźż", "acelnoszz")


def _candidates_pl(word: str) -> set[str]:
    def part(p: str) -> set[str]:
        out = {p}
        for end, repl in _DECLENSION:
            # "-ski/-cki/-dzki" is a masculine nominative, not a feminine genitive like
            # "Agnieszki": otherwise Kozłowski -> Kozłowska would match a woman's surname
            if end == "i" and p.endswith(("ski", "cki", "dzki")):
                continue
            if p.endswith(end) and len(p) - len(end) >= 2:
                out.add(p[: -len(end)] + repl)
        return out

    combos = {""}
    for options in (part(p) for p in word.lower().split("-")):
        combos = {f"{c}-{o}" if c else o for c in combos for o in options}
    return combos


@functools.lru_cache(maxsize=50_000)
def base_candidates(word: str) -> frozenset[str]:
    """Nominative candidates of a name, with and without Polish diacritics.

    Students often type "Mandziuk" for "Mańdziuk", so both sides are compared in
    both spellings. Cached: the same words come back with every request.
    """
    with_pl = _candidates_pl(word)
    plain = _candidates_pl(word.lower().translate(_NO_DIACRITICS))
    return frozenset(with_pl | {c.translate(_NO_DIACRITICS) for c in with_pl} | plain)


@functools.lru_cache(maxsize=10_000)
def inflected_forms(word: str) -> frozenset[str]:
    """All expected inflected forms of a nominative name (both spellings)."""
    w = word.lower()
    forms = {w}
    for nom, ends in _INFLECT.items():
        if w.endswith(nom):
            forms |= {w[: -len(nom)] + e for e in ends}
    if w.endswith("a"):
        forms |= {w[:-1] + e for e in ("y", "i", "ie", "ą", "ę", "ce", "rze", "dze")}
    elif w[-1:] not in "aeiouyąęó":
        forms |= {w + e for e in ("a", "owi", "em", "iem", "u", "ie")}
        if w.endswith("ek"):
            forms |= {w[:-2] + e for e in ("ka", "kowi", "kiem")}
    return frozenset(forms | {f.translate(_NO_DIACRITICS) for f in forms})


def same_name(a: str, b: str) -> bool:
    return bool(base_candidates(a) & base_candidates(b))


def one_typo(a: str, b: str) -> bool:
    """True if a and b differ by one keyboard typo: an extra, missing or wrong
    letter, or two neighbouring letters swapped ("Lewandowksi").

    A different LAST letter does not count: "Kozłowski"/"Kozłowska" or
    "Górny"/"Górna" are different people, not a typo. An extra last letter
    counts only when it doubles the previous one ("Rybakk").
    """
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return diff[0] != len(a) - 1
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and a[diff[0]] == b[diff[1]]
            and a[diff[1]] == b[diff[0]]
        )
    short, long_ = sorted((a, b), key=len)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1 :] == short:
            return i < len(long_) - 1 or long_[-1] == long_[-2]
    return False


def similar_first_name(a: str, b: str) -> bool:
    """Same first name up to declension/diacritics, or one typo away (4+ letters)."""
    if same_name(a, b):
        return True
    return any(
        len(x) >= 4 and len(y) >= 4 and one_typo(x, y)
        for x in base_candidates(a)
        for y in base_candidates(b)
    )


_NOT_A_NAME_WORD = {  # prepositions etc. a model span may start with ("z Bogucką")
    "z",
    "ze",
    "w",
    "we",
    "u",
    "o",
    "od",
    "do",
    "na",
    "dla",
    "przy",
    "pod",
    "nad",
    "za",
    "i",
    "a",
    "oraz",
    "of",
    "at",
    "with",
    "from",
    "for",
    "to",
    "and",
    "з",
    "зі",
    "у",
    "в",
    "до",
    "на",
    "від",
    "для",
    "і",
}


def name_words(text: str) -> list[str]:
    return [
        w
        for w in _NAME_WORD.findall(text)
        if len(w) > 1 and w.lower() not in _NOT_A_NAME_WORD and not _TITLE_WORD.match(w)
    ]


@dataclass(frozen=True)
class StaffEntry:
    surname: str  # as written in the source (URL: lowercase, no diacritics)
    first: str | None = None


def parse_staff_urls(urls) -> list[StaffEntry]:
    """``.../pracownicy/prac_<surname>_<first>/`` -> StaffEntry (surname, first)."""
    out: dict[tuple[str, str], StaffEntry] = {}
    for url in urls:
        m = _STAFF_URL.search(url.strip())
        if m:
            out.setdefault((m.group(1), m.group(2)), StaffEntry(m.group(1), m.group(2)))
    return list(out.values())


_UP, _LOW = "A-ZĄĆĘŁŃÓŚŹŻ", "a-ząćęłńóśźż"
_NAME = rf"[{_UP}][{_LOW}]+(?:-[{_UP}][{_LOW}]+)?"
_TITLES = r"(?:prof\.?|dr\.?|hab\.?|inż\.?|mgr\.?|doc\.?|lic\.?)"
_ROLES = r"(?i:rektor\w*|prorektor\w*|dziekan\w*|prodziekan\w*|kierowni\w*|pełnomocni\w*|opiekun\w*|koordynator\w*|sekretar\w*)"
# "prof. dr hab. inż. Jan Kowalski", "Kierownik dziekanatu: Anna Nowak", "Prodziekan ds. Studenckich – Jan Kowalski"
_TITLED_NAME = re.compile(
    rf"(?<!\w)(?:(?:{_TITLES}\s+)+|{_ROLES}[^:–\n]{{0,60}}?\s*[:–]\s*)((?:{_NAME}\s+){{1,2}}{_NAME})"
)


def parse_staff_text(texts) -> list[StaffEntry]:
    """Names after an academic title / a function with a separator, in any grammatical case.

    A three-word name gives two entries, so both "Anna Kowalska" and
    "Maria Kowalska" match "Anna Maria Kowalska".
    """
    out: dict[tuple[str, str], StaffEntry] = {}
    for text in texts:
        for m in _TITLED_NAME.finditer(text or ""):
            words = m.group(1).split()
            for first in words[:-1]:
                out.setdefault(
                    (first.lower(), words[-1].lower()), StaffEntry(words[-1], first)
                )
    return list(out.values())


def load_staff_from_knowledge_base() -> list[StaffEntry]:
    """Names from the texts stored in Qdrant (the chatbot's knowledge base). Never raises."""
    try:
        from src.api.retrieval import (
            _get_qdrant_client,  # noqa: PLC0415  loads the retrieval models
        )
        from src.ingestion.vector_db import COLLECTION_NAME  # noqa: PLC0415

        client = _get_qdrant_client()
        entries: list[StaffEntry] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=1000,
                with_payload=["text"],
                with_vectors=False,
            )
            entries += parse_staff_text(
                (p.payload or {}).get("text", "") for p in points
            )
            if offset is None:
                break
        logger.info("staff list: %d names found in the knowledge base", len(entries))
        return entries
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "staff list: could not read the knowledge base (%s)", type(exc).__name__
        )
        return []


def parse_staff_lines(lines) -> list[StaffEntry]:
    """Extra list file: "Imię Nazwisko" or "Nazwisko" per line, "#" comments."""
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = name_words(line)
        if words:
            out.append(StaffEntry(words[-1], words[-2] if len(words) >= 2 else None))
    return out


class StaffList:
    def __init__(self, entries=()) -> None:
        self.entries: list[StaffEntry] = []
        self._by_surname: dict[str, list[StaffEntry]] = {}
        self.add(entries)

    def add(self, entries) -> None:
        """Add entries (duplicates skipped); safe to call while requests are served."""
        known = {(e.first or "", e.surname) for e in self.entries}
        for e in entries:
            if (e.first or "", e.surname) in known:
                continue
            known.add((e.first or "", e.surname))
            self.entries.append(e)
            for cand in base_candidates(e.surname):
                self._by_surname.setdefault(cand, []).append(e)

    def __len__(self) -> int:
        return len(self.entries)

    def is_staff(self, span: str) -> bool:
        """True if the span names a person from the list (see module docstring)."""
        words = name_words(span)
        if not words:
            return False
        surname, first = words[-1], (words[-2] if len(words) >= 2 else None)
        # 1. surname matches up to declension/diacritics; first names must agree when both are known
        for cand in base_candidates(surname):
            for e in self._by_surname.get(cand, ()):
                if (
                    first is None
                    or e.first is None
                    or similar_first_name(first, e.first)
                ):
                    return True
        # 2. one typo in the surname ("Lewandowksi"): only with a matching first name, never for a bare surname
        if first:
            mine = [c for c in base_candidates(surname) if len(c) >= 5]
            for e in self.entries:
                if (
                    e.first
                    and similar_first_name(first, e.first)
                    and any(
                        one_typo(c, f)
                        for c in mine
                        for f in inflected_forms(e.surname)
                        if len(f) >= 5
                    )
                ):
                    return True
        return False


def load_staff_list(extra_file: str | Path | None = None) -> StaffList:
    """Staff pages from the scraping list + the optional extra file. Never raises."""
    entries: list[StaffEntry] = []
    try:
        from src.ingestion.links_extended import links  # noqa: PLC0415

        entries += parse_staff_urls(links)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "staff list: could not read links_extended (%s)", type(exc).__name__
        )
    if extra_file:
        try:
            entries += parse_staff_lines(
                Path(extra_file).read_text(encoding="utf-8").splitlines()
            )
        except OSError as exc:
            logger.warning(
                "staff list: could not read %s (%s)", extra_file, type(exc).__name__
            )
    staff = StaffList(entries)
    logger.info("staff list: %d entries", len(staff))
    return staff
