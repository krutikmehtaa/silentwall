"""SEC EDGAR retrieval.

Two things shape this module. First, EDGAR needs no API key, so the whole corpus is
public domain and redistributable. Second, SEC asks automated clients to send a
descriptive User-Agent with a contact address and to stay under ten requests per
second, so every call goes through a token bucket and the client refuses to start
without a User-Agent.

Enumeration uses the quarterly full index rather than the search API because the
index is a stable flat file: the same quarter always lists the same filings in the
same order, which is what makes corpus construction reproducible.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..errors import ConfigError, CorpusFetchError
from ..hashing import sha256_hex

__all__ = ["IndexRow", "HttpCache", "DiskHttpCache", "TokenBucket", "EdgarClient"]

EDGAR_BASE = "https://www.sec.gov"
DATA_BASE = "https://data.sec.gov"
FORM_INDEX = EDGAR_BASE + "/Archives/edgar/full-index/{year}/QTR{q}/form.idx"


@dataclass(frozen=True, slots=True)
class IndexRow:
    """One line of a quarterly form index."""

    form_type: str
    company_name: str
    cik: str
    date_filed: str
    file_name: str

    @property
    def accession(self) -> str:
        return Path(self.file_name).stem


class HttpCache(Protocol):
    def get(self, url: str) -> bytes | None: ...
    def put(self, url: str, body: bytes) -> None: ...


class DiskHttpCache:
    """Content-addressed HTTP cache.

    The point is that the second run of corpus construction is fully offline. That
    matters for CI, and it means a reviewer can reproduce the corpus without
    hitting SEC at all.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        digest = sha256_hex(url)
        return self.root / digest[:2] / f"{digest}.bin"

    def get(self, url: str) -> bytes | None:
        p = self._path(url)
        return p.read_bytes() if p.exists() else None

    def put(self, url: str, body: bytes) -> None:
        p = self._path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(p)


class MemoryHttpCache:
    """In-memory cache, used by tests and by fixture-backed corpus builds."""

    def __init__(self, seed: dict[str, bytes] | None = None) -> None:
        self._store: dict[str, bytes] = dict(seed or {})

    def get(self, url: str) -> bytes | None:
        return self._store.get(url)

    def put(self, url: str, body: bytes) -> None:
        self._store[url] = body


class TokenBucket:
    """Simple rate limiter. Refills at a fixed rate, must be decremented first."""

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_sec)
        self._tokens = self.capacity
        self._last = time.monotonic()

    def take(self, amount: float = 1.0) -> None:
        while True:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= amount:
                self._tokens -= amount
                return
            time.sleep((amount - self._tokens) / self.rate)


class EdgarClient:
    """Keyless EDGAR access with rate limiting, retries and a read-through cache."""

    def __init__(
        self,
        user_agent: str,
        cache: HttpCache,
        rps: float = 8.0,
        max_retries: int = 4,
        session: Any | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ConfigError(
                "EdgarClient needs a descriptive User-Agent with a contact address, "
                "as SEC asks of automated clients"
            )
        self.user_agent = user_agent
        self.cache = cache
        self.bucket = TokenBucket(rps)
        self.max_retries = max_retries
        self._session = session
        self._rng = random.Random(0)

    def _http(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
            )
        return self._session

    def fetch(self, url: str) -> bytes:
        """GET with cache, rate limit, and backoff on 429 or 5xx."""
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        last_detail = ""
        for attempt in range(self.max_retries):
            self.bucket.take()
            try:
                resp = self._http().get(url, timeout=30)
            except Exception as exc:  # noqa: BLE001 transport errors are all retryable
                last_detail = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    body: bytes = resp.content
                    self.cache.put(url, body)
                    return body
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_detail = f"HTTP {resp.status_code}"
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(min(30.0, float(retry_after)))
                            continue
                        except ValueError:
                            pass
                else:
                    raise CorpusFetchError(f"{url} returned HTTP {resp.status_code}")

            # exponential backoff with jitter so parallel clients do not sync up
            time.sleep(min(30.0, (2**attempt) * 0.5 + self._rng.random() * 0.5))

        raise CorpusFetchError(f"{url} failed after {self.max_retries} attempts: {last_detail}")

    def form_index(self, quarter: str, form_types: tuple[str, ...] = ()) -> Iterator[IndexRow]:
        """Yield index rows for a quarter like '2019Q1', optionally filtered by form."""
        year, q = _parse_quarter(quarter)
        raw = self.fetch(FORM_INDEX.format(year=year, q=q))
        wanted = {f.upper() for f in form_types}

        text = raw.decode("latin-1", errors="replace")
        started = False
        for line in text.splitlines():
            if not started:
                # the header block ends with a run of dashes
                if set(line.strip()) == {"-"} and len(line.strip()) > 10:
                    started = True
                continue
            row = _parse_index_line(line)
            if row is None:
                continue
            if wanted and row.form_type.upper() not in wanted:
                continue
            yield row

    def company_facts(self, cik: str) -> dict[str, Any]:
        """XBRL company facts, used for the size band in control matching."""
        import json

        padded = str(cik).lstrip("0").zfill(10)
        url = f"{DATA_BASE}/api/xbrl/companyfacts/CIK{padded}.json"
        try:
            return dict(json.loads(self.fetch(url)))
        except CorpusFetchError:
            return {}


def _parse_quarter(quarter: str) -> tuple[int, int]:
    q = quarter.upper().strip()
    if len(q) != 6 or q[4] != "Q" or not q[:4].isdigit() or q[5] not in "1234":
        raise ConfigError(f"quarter must look like '2019Q1', got {quarter!r}")
    return int(q[:4]), int(q[5])


def _parse_index_line(line: str) -> IndexRow | None:
    """form.idx is fixed-width. Splitting on runs of whitespace is close enough
    because the file name column never contains spaces."""
    if not line.strip():
        return None
    parts = line.split()
    if len(parts) < 4 or not parts[-1].endswith(".txt"):
        return None
    file_name = parts[-1]
    date_filed = parts[-2]
    cik = parts[-3]
    if not cik.isdigit() or len(date_filed) != 10:
        return None
    form_type = parts[0]
    company_name = " ".join(parts[1:-3])
    return IndexRow(
        form_type=form_type,
        company_name=company_name,
        cik=cik,
        date_filed=date_filed,
        file_name=file_name,
    )
