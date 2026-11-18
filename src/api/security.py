"""Abuse protection for the chatbot API.

Pure-Python building blocks without ML or network imports, so they can be unit
tested quickly: settings, client address detection, rate limiting, the cost
budget and JSON security-event logging.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

CHARS_PER_TOKEN = 3.0  # rough estimate for Polish text, not measured
PROMPT_OVERHEAD_CHARS = 20_000  # system prompt + retrieved facts + FAQ (estimate)
DEFAULT_MAX_TOKENS = 1024  # same default as src/api/main.py
MINUTE_SEC = 60.0
HOUR_SEC = 3600.0
_PURGE_THRESHOLD = 10_000  # purge idle per-client state above this many keys


@dataclass(frozen=True)
class Settings:
    """All tunable limits; the defaults are starting points to be calibrated."""

    query_max_chars: int = 1000
    max_tokens_cap: int = 2048
    rate_limit_per_min: int = 20
    rate_limit_per_hour: int = 150
    budget_per_ip_usd_hour: float = 0.50
    budget_global_usd_hour: float = 10.0
    budget_warn_ratio: float = 0.8
    trusted_proxy_hops: int = 2
    internal_api_key: str = ""
    retrieval_top_k_max: int = 50
    enable_docs: bool = False
    model_allowlist: tuple[str, ...] = (
        "google/gemini-2.5-flash",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat-v3-0324",
        "mistralai/mistral-small-3.2-24b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
    )
    default_model: str = "deepseek/deepseek-chat-v3-0324"
    cost_input_usd_per_mtok: float = 0.29  # deepseek-chat-v3-0324, to be confirmed
    cost_output_usd_per_mtok: float = 1.14
    cost_overhead_usd_per_request: float = 0.0005  # translator + query rewriter
    security_log_throttle_sec: float = 60.0

    def public_dict(self) -> dict[str, Any]:
        """
        Return the settings without secrets, for the startup log line.

        Returns
        -------
        dict[str, Any]
            All settings; the internal key is replaced by a boolean flag.
        """
        data = asdict(self)
        key = data.pop("internal_api_key")
        data["internal_key_configured"] = bool(key)
        return data


def _read(
    env: Mapping[str, str], name: str, cast: Callable[[str], Any], default: Any
) -> Any:
    """Read ``name`` from ``env`` and cast it; use ``default`` if unset or invalid."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        return default


def _as_bool(raw: str) -> bool:
    """Interpret common truthy spellings of an environment variable."""
    return raw.lower() in {"1", "true", "yes", "on"}


def _read_list(
    env: Mapping[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    """Read a comma-separated list from ``env``; use ``default`` if it is empty."""
    parts = env.get(name, "").split(",")
    items = tuple(part.strip() for part in parts if part.strip())
    return items or default


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """
    Build `Settings` from environment variables.

    Parameters
    ----------
    env : Mapping[str, str] | None
        Variables to read; defaults to ``os.environ``.

    Returns
    -------
    Settings
        Parsed settings; unset or invalid values fall back to the defaults.
    """
    env = os.environ if env is None else env
    d = Settings()
    return Settings(
        query_max_chars=_read(env, "QUERY_MAX_CHARS", int, d.query_max_chars),
        max_tokens_cap=_read(env, "MAX_TOKENS_CAP", int, d.max_tokens_cap),
        rate_limit_per_min=_read(env, "RATE_LIMIT_PER_MIN", int, d.rate_limit_per_min),
        rate_limit_per_hour=_read(
            env, "RATE_LIMIT_PER_HOUR", int, d.rate_limit_per_hour
        ),
        budget_per_ip_usd_hour=_read(
            env, "BUDGET_PER_IP_USD_HOUR", float, d.budget_per_ip_usd_hour
        ),
        budget_global_usd_hour=_read(
            env, "BUDGET_GLOBAL_USD_HOUR", float, d.budget_global_usd_hour
        ),
        budget_warn_ratio=_read(env, "BUDGET_WARN_RATIO", float, d.budget_warn_ratio),
        trusted_proxy_hops=_read(env, "TRUSTED_PROXY_HOPS", int, d.trusted_proxy_hops),
        internal_api_key=env.get("INTERNAL_API_KEY", d.internal_api_key),
        retrieval_top_k_max=_read(
            env, "RETRIEVAL_TOP_K_MAX", int, d.retrieval_top_k_max
        ),
        enable_docs=_read(env, "ENABLE_DOCS", _as_bool, d.enable_docs),
        model_allowlist=_read_list(env, "MODEL_ALLOWLIST", d.model_allowlist),
        default_model=env.get("DEFAULT_MODEL", "").strip() or d.default_model,
        cost_input_usd_per_mtok=_read(
            env, "COST_INPUT_USD_PER_MTOK", float, d.cost_input_usd_per_mtok
        ),
        cost_output_usd_per_mtok=_read(
            env, "COST_OUTPUT_USD_PER_MTOK", float, d.cost_output_usd_per_mtok
        ),
        cost_overhead_usd_per_request=_read(
            env,
            "COST_OVERHEAD_USD_PER_REQUEST",
            float,
            d.cost_overhead_usd_per_request,
        ),
        security_log_throttle_sec=_read(
            env, "SECURITY_LOG_THROTTLE_SEC", float, d.security_log_throttle_sec
        ),
    )


def truncate_query(text: str, max_chars: int) -> str:
    """
    Cut ``text`` to at most ``max_chars`` characters (characters, not bytes).

    Parameters
    ----------
    text : str
        User query.
    max_chars : int
        Maximum length.

    Returns
    -------
    str
        ``text`` unchanged if short enough, otherwise its first ``max_chars`` characters.
    """
    return text if len(text) <= max_chars else text[:max_chars]


def clamp_max_tokens(value: Any, cap: int) -> int:
    """
    Limit a client-supplied ``max_tokens`` to ``cap``.

    Parameters
    ----------
    value : Any
        Value received from the client (may be missing, a string or garbage).
    cap : int
        Server-side ceiling.

    Returns
    -------
    int
        ``value`` as int if it is a positive number not above ``cap``, else ``cap``.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return cap
    if number <= 0:
        return cap
    return min(number, cap)


def verify_internal_key(provided: str | None, expected: str) -> bool:
    """
    Compare the ``X-Internal-Key`` header with the configured key in constant time.

    Parameters
    ----------
    provided : str | None
        Header value sent by the caller.
    expected : str
        Configured key; an empty key means "not configured" and never matches.

    Returns
    -------
    bool
        True only when a key is configured and the caller sent the same one.
    """
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


def messages_chars(messages: list[dict[str, Any]]) -> int:
    """
    Count the characters of text sent to the model (attachments are not counted).

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Messages array in the OpenAI chat format.

    Returns
    -------
    int
        Total length of all string contents and of ``text`` parts.
    """
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(str(part.get("text", "")))
    return total


def input_cost_usd(chars: float, settings: Settings) -> float:
    """Estimated cost in USD of sending ``chars`` characters to the model."""
    tokens = chars / CHARS_PER_TOKEN
    return tokens * settings.cost_input_usd_per_mtok / 1_000_000


def output_cost_usd(tokens: float, settings: Settings) -> float:
    """Estimated cost in USD of ``tokens`` generated tokens."""
    return tokens * settings.cost_output_usd_per_mtok / 1_000_000


def reserved_cost_usd(input_chars: float, max_tokens: int, settings: Settings) -> float:
    """
    Upper bound of one request's cost, reserved before the model is called.

    Assumes the answer uses the whole ``max_tokens`` allowance and adds a flat
    amount for the translator and the query rewriter.
    """
    return (
        input_cost_usd(input_chars, settings)
        + output_cost_usd(max_tokens, settings)
        + settings.cost_overhead_usd_per_request
    )


def actual_cost_usd(
    input_chars: float, output_chars: float, settings: Settings
) -> float:
    """Estimated real cost of a finished request (used to settle a reservation)."""
    return (
        input_cost_usd(input_chars, settings)
        + output_cost_usd(output_chars / CHARS_PER_TOKEN, settings)
        + settings.cost_overhead_usd_per_request
    )


@dataclass(frozen=True)
class ClientId:
    """Who is calling: the address used for limits and whether it is internal."""

    ip: str
    internal: bool


def identify_client(
    headers: Mapping[str, str], peer: str | None, trusted_hops: int
) -> ClientId:
    """
    Work out the caller's address behind the reverse proxies.

    Every trusted proxy appends the address of *its* peer to
    ``X-Forwarded-For``, so the real client is the ``trusted_hops``-th entry
    counted from the right. Entries further left can be forged by the caller
    and are ignored. Traffic without the header never passed the public proxy
    chain (for example our scripts inside the Docker network) and is treated
    as internal.

    Parameters
    ----------
    headers : Mapping[str, str]
        Request headers (names are matched case-insensitively).
    peer : str | None
        Address of the direct TCP peer.
    trusted_hops : int
        Number of trusted proxies that append to the header (Apache + Vite = 2).

    Returns
    -------
    ClientId
        The detected address and whether the call is internal.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    raw = lowered.get("x-forwarded-for", "")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        return ClientId(ip=peer or "unknown", internal=True)
    index = max(len(entries) - max(trusted_hops, 1), 0)
    return ClientId(ip=entries[index], internal=False)


@dataclass(frozen=True)
class RateDecision:
    """Result of a rate-limit check."""

    allowed: bool
    retry_after: int = 0  # seconds, rounded up
    window: str = ""  # "minute" or "hour" when denied


class RateLimiter:
    """
    Sliding-window limiter per client key: N requests per minute and per hour.

    Only allowed requests are recorded, so a blocked client is released as soon
    as its oldest request leaves the window.
    """

    def __init__(
        self,
        per_min: int,
        per_hour: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_min = per_min
        self._per_hour = per_hour
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> RateDecision:
        """
        Register one request from ``key`` if the limits allow it.

        Parameters
        ----------
        key : str
            Client address.

        Returns
        -------
        RateDecision
            ``allowed=True`` (and the request is recorded) or the reason and
            the number of seconds to wait.
        """
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - HOUR_SEC:
                hits.popleft()
            last_minute = [stamp for stamp in hits if stamp > now - MINUTE_SEC]
            if len(last_minute) >= self._per_min:
                wait = last_minute[0] + MINUTE_SEC - now
                return RateDecision(False, max(1, math.ceil(wait)), "minute")
            if len(hits) >= self._per_hour:
                wait = hits[0] + HOUR_SEC - now
                return RateDecision(False, max(1, math.ceil(wait)), "hour")
            hits.append(now)
            if len(self._hits) > _PURGE_THRESHOLD:
                self._purge(now)
            return RateDecision(True)

    def _purge(self, now: float) -> None:
        """Forget clients with no request in the last hour (caller holds the lock)."""
        stale = [k for k, h in self._hits.items() if not h or h[-1] <= now - HOUR_SEC]
        for key in stale:
            del self._hits[key]


@dataclass(frozen=True)
class BudgetDecision:
    """Result of a budget reservation attempt."""

    allowed: bool
    scope: str = ""  # "ip" or "global" when denied
    warn: bool = False  # this reservation crossed the global warning threshold
    retry_after: int = 0  # seconds until the oldest spend leaves the window


@dataclass
class Reservation:
    """A reserved (and later settled) amount of money, kept for one hour."""

    stamp: float
    ip: str
    usd: float
    live: bool = True  # False once the reservation left the window


class BudgetTracker:
    """
    Rolling one-hour spend per client address and globally.

    A request first *reserves* an upper-bound cost (so parallel requests cannot
    overshoot the limit) and afterwards *settles* the reservation with the
    estimated real cost.
    """

    def __init__(
        self,
        per_ip_usd: float,
        global_usd: float,
        warn_ratio: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_ip = per_ip_usd
        self._global = global_usd
        self._warn_ratio = warn_ratio
        self._clock = clock
        self._items: deque[Reservation] = deque()
        self._ip_total: defaultdict[str, float] = defaultdict(float)
        self._global_total = 0.0
        self._lock = threading.Lock()

    def reserve(self, ip: str, usd: float) -> tuple[BudgetDecision, Reservation | None]:
        """
        Try to reserve ``usd`` for ``ip``.

        Parameters
        ----------
        ip : str
            Client address.
        usd : float
            Upper-bound cost of the request.

        Returns
        -------
        tuple[BudgetDecision, Reservation | None]
            The decision and, when allowed, the reservation to settle later.
        """
        now = self._clock()
        with self._lock:
            self._prune(now)
            if self._ip_total.get(ip, 0.0) + usd > self._per_ip:
                return (
                    BudgetDecision(False, "ip", retry_after=self._wait(now, ip)),
                    None,
                )
            if self._global_total + usd > self._global:
                return (
                    BudgetDecision(False, "global", retry_after=self._wait(now)),
                    None,
                )
            before = self._global_total
            reservation = Reservation(now, ip, usd)
            self._items.append(reservation)
            self._ip_total[ip] += usd
            self._global_total += usd
            threshold = self._warn_ratio * self._global
            warn = before < threshold <= self._global_total
            return BudgetDecision(True, warn=warn), reservation

    def settle(self, reservation: Reservation, actual_usd: float) -> None:
        """
        Replace a reservation's amount with the estimated real cost.

        Parameters
        ----------
        reservation : Reservation
            Value returned by `reserve`.
        actual_usd : float
            Estimated real cost; ignored if the reservation already expired.
        """
        with self._lock:
            if not reservation.live:
                return
            delta = actual_usd - reservation.usd
            reservation.usd = actual_usd
            self._ip_total[reservation.ip] += delta
            self._global_total += delta

    def _prune(self, now: float) -> None:
        """Drop reservations older than one hour (caller holds the lock)."""
        while self._items and self._items[0].stamp <= now - HOUR_SEC:
            old = self._items.popleft()
            old.live = False
            self._ip_total[old.ip] -= old.usd
            self._global_total -= old.usd
            if self._ip_total[old.ip] <= 1e-12:
                del self._ip_total[old.ip]

    def _wait(self, now: float, ip: str | None = None) -> int:
        """Seconds until the oldest matching reservation expires."""
        for item in self._items:
            if ip is None or item.ip == ip:
                return max(1, math.ceil(item.stamp + HOUR_SEC - now))
        return 1


class SecurityEvents:
    """
    Writes security events as one JSON line each to the ``security`` logger.

    The logger propagates to the root logger, so events leave the container the
    same way as all other application logs (stderr). Repeated events of the
    same type from the same address are throttled: the first one is written at
    once, the rest are only counted, and the count is attached to the next
    line written after the window.
    """

    def __init__(
        self,
        throttle_sec: float,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._throttle = throttle_sec
        self._clock = clock
        self._logger = logger or logging.getLogger("security")
        self._state: dict[tuple[str, str], list[float]] = {}  # key -> [last, dropped]
        self._lock = threading.Lock()

    def emit(
        self, event: str, level: int, ip: str, endpoint: str, **details: Any
    ) -> None:
        """
        Log one event unless an identical one was logged very recently.

        Parameters
        ----------
        event : str
            Event type, e.g. ``rate_limit_minute``.
        level : int
            ``logging`` level (WARNING for violations, ERROR for exhausted budget).
        ip : str
            Client address.
        endpoint : str
            Request path.
        **details : Any
            Extra JSON-serialisable fields. Never pass user text or secrets.
        """
        now = self._clock()
        key = (event, ip)
        with self._lock:
            state = self._state.get(key)
            if state is not None and now - state[0] < self._throttle:
                state[1] += 1
                return
            dropped = int(state[1]) if state else 0
            self._state[key] = [now, 0.0]
            if len(self._state) > _PURGE_THRESHOLD:
                self._purge(now)
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": event,
            "ip": ip,
            "endpoint": endpoint,
            **details,
        }
        if dropped:
            payload["suppressed"] = dropped
        self._logger.log(level, json.dumps(payload, ensure_ascii=False, default=str))

    def log_startup(self, settings: Settings) -> None:
        """Log the active limits once at application start (no secrets)."""
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": "security_startup",
            **settings.public_dict(),
        }
        self._logger.info(json.dumps(payload, ensure_ascii=False))

    def _purge(self, now: float) -> None:
        """Forget throttle state older than the window (caller holds the lock)."""
        stale = [k for k, s in self._state.items() if now - s[0] >= self._throttle]
        for key in stale:
            del self._state[key]


class RateLimited(Exception):
    """Raised when a client exceeded its request rate."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


class BudgetExceeded(Exception):
    """Raised when the cost budget of a client (``ip``) or of everyone (``global``) is used up."""

    def __init__(self, scope: str, retry_after: int) -> None:
        super().__init__(f"{scope} budget exceeded, retry after {retry_after}s")
        self.scope = scope
        self.retry_after = retry_after


class SecurityGuard:
    """
    Single entry point used by the API: limits, budget and event logging.

    Internal traffic (no ``X-Forwarded-For``) is exempt from the rate limit and
    from the budget.
    """

    def __init__(
        self,
        settings: Settings,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.events = SecurityEvents(settings.security_log_throttle_sec, clock, logger)
        self._rate = RateLimiter(
            settings.rate_limit_per_min, settings.rate_limit_per_hour, clock
        )
        self._budget = BudgetTracker(
            settings.budget_per_ip_usd_hour,
            settings.budget_global_usd_hour,
            settings.budget_warn_ratio,
            clock,
        )

    def identify(self, headers: Mapping[str, str], peer: str | None) -> ClientId:
        """Detect the caller (see `identify_client`)."""
        return identify_client(headers, peer, self.settings.trusted_proxy_hops)

    def check_rate(self, client: ClientId, endpoint: str) -> None:
        """
        Enforce the per-address request rate.

        Raises
        ------
        RateLimited
            When the address exceeded the per-minute or per-hour limit.
        """
        if client.internal:
            return
        decision = self._rate.check(client.ip)
        if not decision.allowed:
            self.events.emit(
                f"rate_limit_{decision.window}",
                logging.WARNING,
                client.ip,
                endpoint,
                retry_after=decision.retry_after,
            )
            raise RateLimited(decision.retry_after)

    def limit_query(self, text: str, client: ClientId, endpoint: str) -> str:
        """Truncate the query to the configured length and log when it was cut."""
        limited = truncate_query(text, self.settings.query_max_chars)
        if len(limited) != len(text):
            self.events.emit(
                "query_truncated",
                logging.WARNING,
                client.ip,
                endpoint,
                original_chars=len(text),
                limit=self.settings.query_max_chars,
            )
        return limited

    def limit_top_k(self, value: int, client: ClientId, endpoint: str) -> int:
        """Clamp ``top_k`` to ``[1, retrieval_top_k_max]`` and log when changed."""
        limited = max(1, min(int(value), self.settings.retrieval_top_k_max))
        if limited != value:
            self.events.emit(
                "top_k_clamped",
                logging.WARNING,
                client.ip,
                endpoint,
                requested=value,
                limit=self.settings.retrieval_top_k_max,
            )
        return limited

    def sanitize_model_config(
        self, config: dict[str, Any] | None, client: ClientId, endpoint: str
    ) -> dict[str, Any] | None:
        """
        Return a copy of the client's model config that is safe to use.

        External clients may only use models from the allowlist (anything else
        is replaced by the default model) and ``max_tokens`` is capped. Internal
        traffic (our own scripts) may choose any model.

        Returns
        -------
        dict[str, Any] | None
            ``config`` itself when nothing had to change, otherwise a fixed copy.
        """
        if not config:
            return config
        clean = dict(config)
        if "model" in clean and not client.internal:
            requested = clean["model"]
            if requested not in self.settings.model_allowlist:
                clean["model"] = self.settings.default_model
                self.events.emit(
                    "model_replaced",
                    logging.WARNING,
                    client.ip,
                    endpoint,
                    requested=str(requested)[:60],
                    used=self.settings.default_model,
                )
        if "max_tokens" in clean:
            capped = clamp_max_tokens(clean["max_tokens"], self.settings.max_tokens_cap)
            if capped != clean["max_tokens"]:
                self.events.emit(
                    "max_tokens_clamped",
                    logging.WARNING,
                    client.ip,
                    endpoint,
                    requested=str(clean["max_tokens"])[:20],
                    limit=capped,
                )
                clean["max_tokens"] = capped
        return config if clean == config else clean

    def reserve(
        self, client: ClientId, endpoint: str, input_chars: int, max_tokens: Any
    ) -> Reservation | None:
        """
        Reserve the worst-case cost of a chat request.

        Returns
        -------
        Reservation | None
            The reservation to pass to `settle`, or None for internal traffic.

        Raises
        ------
        BudgetExceeded
            When the per-address or the global hourly budget would be exceeded.
        """
        if client.internal:
            return None
        tokens = clamp_max_tokens(max_tokens, self.settings.max_tokens_cap)
        amount = reserved_cost_usd(input_chars, tokens, self.settings)
        decision, reservation = self._budget.reserve(client.ip, amount)
        if not decision.allowed:
            exceeded_globally = decision.scope == "global"
            self.events.emit(
                "budget_global_exceeded" if exceeded_globally else "budget_ip_exceeded",
                logging.ERROR if exceeded_globally else logging.WARNING,
                client.ip,
                endpoint,
                reserve_usd=round(amount, 6),
            )
            raise BudgetExceeded(decision.scope, decision.retry_after)
        if decision.warn:
            self.events.emit(
                "budget_global_warn",
                logging.WARNING,
                client.ip,
                endpoint,
                ratio=self.settings.budget_warn_ratio,
            )
        return reservation

    def settle(
        self, reservation: Reservation | None, input_chars: int, output_chars: int
    ) -> None:
        """Replace the reserved amount with the estimated real cost."""
        if reservation is None:
            return
        actual = actual_cost_usd(input_chars, output_chars, self.settings)
        self._budget.settle(reservation, actual)

    def internal_key_ok(
        self, provided: str | None, client: ClientId, endpoint: str
    ) -> bool:
        """Check the ``X-Internal-Key`` header and log a denial."""
        ok = verify_internal_key(provided, self.settings.internal_api_key)
        if not ok:
            self.events.emit(
                "internal_key_denied", logging.WARNING, client.ip, endpoint
            )
        return ok
