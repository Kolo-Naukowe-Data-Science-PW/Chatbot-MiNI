"""Anonymization of user prompts before they leave the VM (Chatbot-MiNI).

Usage in the API:

    anonymizer = Anonymizer()            # once, at startup (config from env)
    anonymizer.warm_up()                 # loads the model, so the first user does not wait
    query = anonymizer.anonymize(query).text

Call it before logging, translation, query rewriting or any request to OpenRouter.

Two layers:
1. regex (`mask_identifiers`): identifiers by FORMAT (PESEL, phone, e-mail,
   account, student ID, first-person address...). Deterministic, ~0 ms.
   RECALL FIRST: no checksum, a fake or mistyped PESEL is masked too.
2. local NER model (GLiNER, inside the API process on CPU, ~0.7 s on the VM):
   names of people. The model only marks fragments; the code replaces them, so
   the model can never rewrite the question. Kept: names with an academic title
   ("dr X"), people the chatbot knows (src/api/staff_names.py), patrons.

Long texts are split into chunks of CHUNK_CHARS characters (the model has a
token limit); more than MAX_CHUNKS chunks is "too long". All rules run on every
text, whatever the UI language (PL / EN / UA).

Configuration (ANONYMIZER_* environment variables) and limitations:
src/api/ANONYMIZATION.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from src.api.staff_names import StaffList, load_staff_list

logger = logging.getLogger(__name__)

# ==========================================================================
# 1. regex: identifiers
# ==========================================================================


@dataclass(frozen=True)
class Rule:
    label: str
    pattern: re.Pattern[str]
    group: int = 0  # 0 = replace whole match, n = replace only capture group n


_L = r"(?<![\w@./-])"  # left boundary: not glued to a word, email, path or number
_R = r"(?![\w@]|[./-]\d)"  # right boundary
_DATE = (
    r"\d{1,2}\s?[./-]\s?\d{1,2}\s?[./-]\s?\d{2,4}"
    r"|\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia"
    r"|września|października|listopada|grudnia)\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня"
    r"|листопада|грудня)\s+\d{4}"
)
# after "street + number": optional flat number and postal code + city
_ADDR_TAIL = (
    r"(?:\s*(?:/|\s(?:m|lok|mieszkania)\.?\s*)\d+[a-z]?)?"
    r"(?:,?\s*\d{2}-\d{3}(?:\s+[^\n,?!.\d]{2,30}?(?=\s*(?:[,?!\n.]|$)))?)?"
)

# Order matters: specific/long patterns before short/general ones.
RULES: list[Rule] = [
    Rule("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    Rule("IP", re.compile(_L + r"(?:\d{1,3}\.){3}\d{1,3}" + _R)),
    # secrets typed by the user: "hasło do USOS to abc123", "PIN: 1234"
    Rule(
        "HASLO",
        re.compile(
            r"(?i)\b(?:hasł\w*|haslo|password|passwd|pin|пароль\w*)\b[^\n:=]{0,25}?"
            r"(?:\bto\b|\bis\b|\bце\b|:|=)\s*(\S+)"
        ),
        group=1,
    ),
    # IBAN / 26-digit Polish account number, plain or grouped by 4
    Rule(
        "NR_KONTA", re.compile(_L + r"(?:[A-Z]{2}\s?)?\d{2}(?:\s?\d{4}){6}" + _R, re.I)
    ),
    # payment card: 13-19 digits, plain or grouped by 4
    Rule("NR_KARTY", re.compile(_L + r"(?:\d{4}[\s-]){3}\d{1,7}" + _R)),
    Rule(
        "DATA_URODZENIA",
        re.compile(
            r"(?i)\b(?:urodz\w*|ur\.|data urodzenia|dat[aęy] ur\.?|born|date of birth|dob|birthday"
            r"|народи\w*|дата народження)\D{0,15}?(" + _DATE + ")"
        ),
        group=1,
    ),
    Rule("PESEL", re.compile(_L + r"\d{11}" + _R)),
    # PESEL with spaces/dashes ("0227 0812 345") only next to the word PESEL:
    # "48 600 123 456" (phone with country code) is 11 digits too
    Rule(
        "PESEL",
        re.compile(
            r"(?i)\bpesel\w*\b[\s:.-]{0,5}(?:to\s+|nr\.?\s?|numer\s+)?(\d(?:[\s-]?\d){10})(?![\s-]?\d)"
        ),
        group=1,
    ),
    Rule(
        "NIP",
        re.compile(
            _L + r"(?:\d{3}-\d{3}-\d{2}-\d{2}|\d{3}-\d{2}-\d{2}-\d{3}|\d{10})" + _R
        ),
    ),
    Rule(
        "TELEFON",
        re.compile(
            _L
            + r"(?:(?:\+|00)\s?48[\s-]?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{3}|\d{2}[\s-]\d{3}[\s-]\d{2}[\s-]\d{2})"
            + _R
        ),
    ),
    # international number ("+380 67 123 4567", "0044 20 7946 0958")
    Rule("TELEFON", re.compile(_L + r"(?:\+|00)\s?\d{1,3}(?:[\s-]?\d){7,12}" + _R)),
    Rule("NR_DOWODU", re.compile(_L + r"[A-Z]{3}\s?\d{6}" + _R, re.I)),
    Rule("NR_PASZPORTU", re.compile(_L + r"[A-Z]{2}\s?\d{7}" + _R, re.I)),
    # old Ukrainian passport: 2 Cyrillic letters + 6 digits ("КВ123456")
    Rule("NR_PASZPORTU", re.compile(r"(?<!\w)[А-ЯІЇЄҐ]{2}\s?\d{6}(?!\w)", re.I)),
    Rule("NR_PRAWA_JAZDY", re.compile(_L + r"\d{5}/\d{2}/\d{4}" + _R)),  # 00123/12/1234
    # student ID: 5-6 digits only next to a keyword (otherwise rooms, codes...), also "31 23 45"
    Rule(
        "NR_ALBUMU",
        re.compile(
            r"(?i)\b(?:album\w*|indeks\w*|legitymacj\w*|student\s+(?:id|number|card)|index\s+number"
            r"|студентськ\w*(?:\s+квитк\w*)?|заліков\w*(?:\s+книжк\w*)?)\b[\s:.-]{0,5}"
            r"(?:to\s+|is\s+|це\s+)?(?:nr\.?\s?|numer\s|no\.?\s?|number\s|№\s?)?"
            r"(\d(?:[\s-]?\d){4,5})(?![\s-]?\d)"
        ),
        group=1,
    ),
    # address in first person only: "mieszkam na Mokotowskiej 12", "mieszkam w Pruszkowie przy Długiej 5",
    # "zameldowany na ...", "na adres ul. ...", "mój adres to Krucza 14 m. 8, 00-500 Warszawa"
    Rule(
        "ADRES",
        re.compile(
            r"(?i)\b(?:mieszkam(?:\s+w\s+[^\n,?!]{2,30}?)?\s+(?:na|przy)|zameldowan[yae]\s+(?:na|w|przy)"
            r"|m[óo]j\s+adres(?:\s+to|\s*:)?|(?:na|pod)\s+adres(?:em)?(?=\s*(?:ul\.|al\.|pl\.|os\.|ulic|alej))"
            r"|adres(?:\s+zamieszkania|\s+korespondencyjny|\s+do\s+korespondencji)?\s*(?::|to\b)"
            r"|i\s+live\s+(?:at|on)|my\s+(?:home\s+)?address(?:\s+is)?\s*:?"
            r"|живу\s+(?:на|за\s+адресою|по)|моя\s+адреса(?:\s+це)?\s*:?)\s*"
            r"(\d+[a-z]?(?:\s?/\s?\d+)?\s+[^\n,?!]{2,40}?(?=\s*(?:[,?!\n]|$))"
            r"|[^\n,?!]{3,60}?\d+[a-z]?(?:\s?/\s?\d+)?" + _ADDR_TAIL + r")"
        ),
        group=1,
    ),
    # dormitory (+ room) in first person: "mieszkam w akademiku Riviera, pokój 312"
    Rule(
        "ADRES",
        re.compile(
            r"(?i)\b(?:mieszkam|zakwaterowan[yae]|jestem)\s+w\s+((?:akademiku|ds|domu\s+studenckim)\s+[^\n,?!\[]{2,30}?"
            r"(?:,?\s*(?:pok(?:ój|oj)?\.?|p\.)\s*\d+\w*)?)(?=\s*(?:[,?!\n.]|$))"
        ),
        group=1,
    ),
    # catch-all: any remaining run of 7+ digits is suspicious
    Rule("NUMER", re.compile(_L + r"\d{7,}" + _R)),
]


@dataclass
class AnonymizationResult:
    text: str
    found: dict[str, int] = field(
        default_factory=dict
    )  # label -> count, never the values
    regex_only: bool = (
        False  # True if the model was unavailable and the fallback was used
    )

    @property
    def changed(self) -> bool:
        return bool(self.found)


def mask_identifiers(text: str) -> AnonymizationResult:
    """Regex layer only: replace identifiers with placeholders like ``[PESEL]``."""
    found: dict[str, int] = {}
    for rule in RULES:

        def _sub(m: re.Match[str], rule: Rule = rule) -> str:
            found[rule.label] = found.get(rule.label, 0) + 1
            if rule.group == 0:
                return f"[{rule.label}]"
            s, e = m.start(rule.group) - m.start(0), m.end(rule.group) - m.start(0)
            return m.group(0)[:s] + f"[{rule.label}]" + m.group(0)[e:]

        text = rule.pattern.sub(_sub, text or "")
    return AnonymizationResult(text, found)


# ==========================================================================
# 2. model layer helpers: which model fragments are NOT private people
# ==========================================================================

# academic titles only: "pan/pani" also precede private people ("moja mama, pani Ewa X")
_TITLE = (
    r"(?:prof\.?|profesor\w*|dr\.?|doktor(?!ant)\w*|hab\.?|inż\.?|mgr\.?|magister\w*)"
)
_TITLE_IN_SPAN = re.compile(rf"(?i)(?<!\w){_TITLE}(?!\w)")
_TITLES_AT_END = re.compile(rf"(?i)(?:(?<!\w){_TITLE}\s*)+$")
# a title right after these words belongs to the AUTHOR: "Jestem inż. Piotr X" (II-degree student)
_INTRO_AT_END = re.compile(
    r"(?i)(?<!\w)(?:jestem|nazywam\s+się|mam\s+na\s+imię|w\s+imieniu|pozdrawiam|z\s+poważaniem"
    r"|tu|mój|moja|moje)\s*[,:–-]?\s*$"
)


def has_title(span: str, text: str) -> bool:
    """True if the span contains an academic title or directly follows one (= staff, keep it).

    Not when the title follows "jestem / nazywam się / mój / moja / w imieniu...":
    then it is the author or someone from their circle.
    """
    for m in re.finditer(re.escape(span.strip()), text, flags=re.I):
        before = text[max(0, m.start() - 60) : m.start()]
        title_before = _TITLES_AT_END.search(before)
        if not (_TITLE_IN_SPAN.search(span) or title_before):
            continue
        if _INTRO_AT_END.search(
            before[: title_before.start()] if title_before else before
        ):
            continue
        return True
    return False


# a name right after these words is a patron / eponym / place, not a person in the chat:
# "sala im. Kopernika", "twierdzenie Banacha", "metoda Newtona", "ul. Noakowskiego"
_PATRON_AT_END = re.compile(
    r"(?i)(?<!\w)(?:im\.|imienia|ul\.|al\.|(?:twierdzeni|metod|rozkład|rozklad|hipotez|algorytm|lemat|wz[oó]r"
    r"|nagrod|stypendi|pomnik|sal|aul|ulic|alej|theorem|method)\w*)\s*$"
)


def is_patron(span: str, text: str) -> bool:
    """True if the span follows a patron / eponym / place word (keep it)."""
    for m in re.finditer(re.escape(span.strip()), text, flags=re.I):
        if _PATRON_AT_END.search(text[max(0, m.start() - 40) : m.start()]):
            return True
    return False


# role words: the model often pulls them into the span ("Pani Anna Nowak", "brat Piotr Nowak",
# "dodatkowego recenzenta"). Short words are listed as whole forms, so "Panek" or "Bratkowski" is not a role.
_ROLE_WORD = re.compile(
    r"(?i)^(?:(?:student|studenc|prodziekan|dziekan|rektor|koleg|koleż|wykładowc|prowadząc|promotor|opiekun"
    r"|pracowni|kierowni|sekretar|kandydat|absolwent|doktorant|pełnomocni|minist|recenzent|nauczyciel"
    r"|interesant|koordynator|członk|uczestni|siostr|rodzic|professor|lecturer|teacher|colleague"
    r"|студент|викладач|декан)\w*"
    r"|osob[aąyęieo]?|osób|pan(?:a|u|em|ie|i|ią|owie|ów|om)?|mam[aąyęo]|tat[aąyęo]|brat(?:a|u|em|owi|ie)?"
    r"|the|my|our|friends?|students|people|mr|mrs|ms|друг|друга)$"
)
_LETTERS = re.compile(r"[^\W\d_]+")


def strip_roles(span: str) -> str:
    """Drop role words in front of a name: "Pani Anna Nowak" -> "Anna Nowak", "rektor Nowak" -> "Nowak"."""
    words = span.split()
    while words and _ROLE_WORD.match(words[0].strip(".,;:")):
        words.pop(0)
    return " ".join(words)


_EXISTING_PLACEHOLDER = re.compile(r"\[[A-ZĄĆĘŁŃÓŚŹŻ_]+\]")
_WORD_CHAR = re.compile(r"[\w-]")


def looks_like_name(span: str) -> bool:
    """Cheap sanity check of a model span (after strip_roles): 1-4 words, no digits, no role word."""
    words = span.split()
    return (
        0 < len(words) <= 4
        and not any(ch.isdigit() for ch in span)
        and not any(_ROLE_WORD.match(w) for w in _LETTERS.findall(span))
    )


def apply_masks(text: str, spans: list[str], placeholder: str = "[OSOBA]") -> str:
    """Replace every occurrence of each span (case-insensitive) with the placeholder.

    Matches are widened to whole words ("Wiśniewski" inside "Wiśniewskiego"),
    matches touching an existing placeholder like "[NR_ALBUMU]" are skipped,
    overlapping matches are merged.
    """
    protected = [m.span() for m in _EXISTING_PLACEHOLDER.finditer(text)]
    intervals: list[tuple[int, int]] = []
    for span in set(spans):
        span = span.strip()
        if len(span) < 2 or "[" in span or "]" in span:
            continue
        for m in re.finditer(re.escape(span), text, flags=re.I):
            start, end = m.span()
            while start > 0 and _WORD_CHAR.match(text[start - 1]):
                start -= 1
            while end < len(text) and _WORD_CHAR.match(text[end]):
                end += 1
            if not any(start < pe and ps < end for ps, pe in protected):
                intervals.append((start, end))
    if not intervals:
        return text
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + placeholder + text[end:]
    return text


# ==========================================================================
# 3. the anonymizer: regex + model + cache + fallback
# ==========================================================================


class TextTooLong(ValueError):
    """More chunks than ANONYMIZER_MAX_CHUNKS: handled like an unavailable model."""


class AnonymizationUnavailable(RuntimeError):
    """The model did not answer and ANONYMIZER_FALLBACK=block: the prompt must not be sent."""


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_into_chunks(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most ``limit`` characters.

    Cuts at line breaks first, then at sentence ends, then at spaces, so a title
    and the name after it ("dr Jan Kowalski") normally stay in one chunk.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    for line in text.splitlines():
        for sentence in _SENTENCE_END.split(line):
            while (
                len(sentence) > limit
            ):  # one very long sentence: cut at the last space
                cut = sentence.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                pieces.append(sentence[:cut])
                sentence = sentence[cut:].lstrip()
            if sentence.strip():
                pieces.append(sentence)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= limit:
            chunks[-1] += "\n" + piece
        else:
            chunks.append(piece)
    return chunks


class Anonymizer:
    LABELS = ["person"]
    THRESHOLD = 0.5  # model confidence; lower = more masked
    CHUNK_CHARS = 1500  # ~250 words, below the model's token limit
    MAX_CHUNKS = 8  # longer question = "too long" (HTTP 413 in the API)
    RETRY_SECONDS = 60  # after a failed model load, try again no sooner than this

    def __init__(
        self,
        model: str | None = None,
        fallback: str | None = None,
        cache_size: int = 1000,
        title_rule: bool = True,
        staff: StaffList | None = None,
    ):
        env = os.environ.get
        self.model_name = model or env(
            "ANONYMIZER_MODEL", "urchade/gliner_multi_pii-v1"
        )
        self.fallback = fallback or env("ANONYMIZER_FALLBACK", "block")
        if self.fallback not in ("block", "regex"):
            raise ValueError(
                f"ANONYMIZER_FALLBACK must be 'block' or 'regex', got {self.fallback!r}"
            )
        self.chunk_chars = self.CHUNK_CHARS
        self.max_chunks = self.MAX_CHUNKS
        self.title_rule = title_rule
        self.staff = (
            staff
            if staff is not None
            else load_staff_list(env("ANONYMIZER_STAFF_FILE"))
        )
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[str, dict[str, int]]] = OrderedDict()
        self._lock = threading.Lock()
        self._model = None
        self._model_lock = threading.Lock()  # loading and inference are serialised
        self._load_failed_at: float | None = None
        self.stats: Counter[str] = (
            Counter()
        )  # model, cache_hit, fallback_regex, blocked, too_long, load_failed

    # ---- model ----------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        """Heavy import only here: the API must start without the model."""
        from gliner import GLiNER  # noqa: PLC0415

        model = GLiNER.from_pretrained(
            self.model_name, cache_dir=os.environ.get("ANONYMIZER_CACHE_DIR")
        )
        model.eval()
        return model

    def _ensure_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model
            if (
                self._load_failed_at
                and time.monotonic() - self._load_failed_at < self.RETRY_SECONDS
            ):
                raise RuntimeError("anonymizer model not loaded")
            try:
                self._model = self._load_model()
            except Exception:
                self._load_failed_at = time.monotonic()
                self.stats["load_failed"] += 1
                raise
            self._load_failed_at = None
            return self._model

    def _detect(self, text: str) -> list[str]:
        """Raw model output: fragments of ``text`` the model marks as a person."""
        model = self._ensure_model()
        with self._model_lock:
            entities = model.predict_entities(
                text, self.LABELS, threshold=self.THRESHOLD
            )
        return [e["text"] for e in entities]

    def _person_alone(self, span: str) -> bool:
        """Does the model mark ``span`` as a person when it stands on its own (without the role word)?"""
        return bool(span) and any(
            e.strip().lower() == span.lower() for e in self._detect(span)
        )

    def _filter(self, spans: list[str], text: str) -> list[str]:
        out = []
        for span in spans:
            span = span.strip()
            rest = strip_roles(span)
            if rest != span and not self._person_alone(rest):
                # "Pani Anna Nowak" -> "Anna Nowak" is a person on its own; "student Wydziału MiNI" is not
                continue
            span = rest
            if (
                "[" in span
                or span.lower() not in text.lower()
                or not looks_like_name(span)
            ):
                continue
            if (
                (self.title_rule and has_title(span, text))
                or is_patron(span, text)
                or self.staff.is_staff(span)
            ):
                continue
            out.append(span)
        return out

    def _model_spans(self, text: str) -> list[str]:
        """Fragments to mask. Raises when the model is unavailable."""
        return self._filter(self._detect(text), text)

    def warm_up(self) -> bool:
        """Load the model and run it once. Call at API startup.

        Never raises: a failed warm-up is logged, and the requests that follow
        use the fallback until the model is up (loading is retried).
        """
        try:
            self._model_spans("Rozgrzewka: jestem Jan Testowy, kiedy jest sesja?")
            logger.info(
                "anonymizer: model %s ready, staff list %d entries",
                self.model_name,
                len(self.staff),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("anonymizer: warm-up failed (%s)", type(exc).__name__)
            return False

    def add_staff(self, entries) -> None:
        """Extend the staff list (names from the knowledge base, at startup) and drop cached results."""
        self.staff.add(entries)
        with self._lock:
            self._cache.clear()

    # ---- public API ------------------------------------------------------

    def anonymize(self, text: str) -> AnonymizationResult:
        """Regex + model. Cached by a hash of the text (the raw text is not kept)."""
        if not text or not text.strip():
            return AnonymizationResult(text or "")
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats["cache_hit"] += 1
                masked, found = self._cache[key]
                return AnonymizationResult(masked, dict(found))

        result = mask_identifiers(text)
        try:
            chunks = split_into_chunks(result.text, self.chunk_chars)
            if len(chunks) > self.max_chunks:
                self.stats["too_long"] += 1
                raise TextTooLong(f"{len(result.text)} chars, {len(chunks)} chunks")
            spans = [s for chunk in chunks for s in self._model_spans(chunk)]
        except (
            Exception
        ) as exc:  # noqa: BLE001  model not loaded, inference error, too long
            if self.fallback == "regex":
                self.stats["fallback_regex"] += 1
                logger.warning(
                    "anonymizer: model unavailable (%s), regex only; stats=%s",
                    type(exc).__name__,
                    dict(self.stats),
                )
                result.regex_only = True
                return result
            self.stats["blocked"] += 1
            logger.error(
                "anonymizer: model unavailable (%s), prompt blocked; stats=%s",
                type(exc).__name__,
                dict(self.stats),
            )
            raise AnonymizationUnavailable("anonymization model unavailable") from exc

        self.stats["model"] += 1
        masked = apply_masks(result.text, spans)
        if masked != result.text:
            result.found["OSOBA"] = masked.count("[OSOBA]") - result.text.count(
                "[OSOBA]"
            )
        result.text = masked
        if self.cache_size:
            with self._lock:
                self._cache[key] = (result.text, dict(result.found))
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return result

    def anonymize_history(self, messages: list) -> list:
        """Anonymize only the user's messages (bot answers were built from masked text).

        Accepts dicts {"role", "content"} or objects with .role/.content (e.g. pydantic).
        The frontend re-sends the whole history with every question, so the cache
        makes earlier messages free.
        """
        out = []
        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            content = msg["content"] if isinstance(msg, dict) else msg.content
            if role != "user":
                out.append(msg)
                continue
            masked = self.anonymize(content).text
            if isinstance(msg, dict):
                out.append({**msg, "content": masked})
            elif hasattr(msg, "model_copy"):
                out.append(msg.model_copy(update={"content": masked}))
            else:
                out.append(type(msg)(role=role, content=masked))
        return out
