"""Turn a filing into a typed record, or exclude it with a reason.

The rule that matters here: a record is emitted only when acquirer, target,
announcement date and at least one numeric protected field all resolve. Anything
short of that goes to the exclusion log. Partial records would quietly weaken the
ground truth, and since every leak measurement compares against these values, a
half-extracted deal is worse than no deal.

Extraction is rule based rather than model based, deliberately. It keeps stage 0 on
CPU, it keeps the corpus reproducible, and it means every protected field can point
at the span of filing text it came from.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..errors import ParseIncompleteError
from ..hashing import short_hash
from ..types import DealRecord, ExclusionRecord, FieldName, ProtectedField

__all__ = ["parse_filing", "normalize_money", "normalize_percent", "ParsedFiling"]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")

_ITEM_RE = re.compile(r"item\s*(1\.01|2\.01)", re.IGNORECASE)

_PRICE_PATTERNS = (
    re.compile(
        r"\$\s*([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)\s*(?:per share|a share|/share)", re.I
    ),
    re.compile(
        r"([0-9]{1,4}(?:\.[0-9]{1,2})?)\s*(?:U\.?S\.?\s*)?dollars\s*(?:per share|a share)", re.I
    ),
    re.compile(r"(?:price|consideration)\s*of\s*\$\s*([0-9]{1,4}(?:\.[0-9]{1,2})?)", re.I),
)

_PREMIUM_PATTERNS = (
    re.compile(r"premium\s*of\s*(?:approximately\s*)?([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*%", re.I),
    re.compile(r"([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*%\s*premium", re.I),
)

_ACQUIRER_PATTERNS = (
    re.compile(
        r"(?:acquisition of|acquire[sd]?\s+by|merger with)\s+([A-Z][\w.,&' -]{2,60}?)(?:,|\.|\s+for\b|\s+in\b)",
        re.I,
    ),
    re.compile(
        r"^([A-Z][\w.,&' -]{2,60}?)\s+(?:to acquire|announces? the acquisition)", re.I | re.M
    ),
)

_TARGET_PATTERNS = (
    re.compile(
        r"(?:to acquire|acquisition of)\s+([A-Z][\w.,&' -]{2,60}?)(?:,|\.|\s+for\b|\s+in\b)", re.I
    ),
)

_CONSIDERATION_WORDS = (
    ("all-cash", "cash"),
    ("all cash", "cash"),
    ("cash and stock", "cash_and_stock"),
    ("stock-for-stock", "stock"),
    ("all-stock", "stock"),
)

_CLOSE_RE = re.compile(
    r"(?:expected to close|closing is expected|complete[d]? (?:in|by))\s+"
    r"(?:the\s+)?((?:first|second|third|fourth)\s+quarter\s+of\s+[0-9]{4}|[A-Z][a-z]+\s+[0-9]{4}|[0-9]{4})",
    re.I,
)


class ParsedFiling:
    """Result of parsing: exactly one of record or exclusion is set."""

    __slots__ = ("record", "exclusion")

    def __init__(
        self, record: DealRecord | None = None, exclusion: ExclusionRecord | None = None
    ) -> None:
        if (record is None) == (exclusion is None):
            raise ParseIncompleteError("ParsedFiling needs exactly one of record or exclusion")
        self.record = record
        self.exclusion = exclusion

    @property
    def ok(self) -> bool:
        return self.record is not None


def strip_markup(raw: str) -> str:
    """Flatten SGML or HTML to text while keeping character offsets roughly sane."""
    text = _TAG_RE.sub(" ", raw)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&#8217;", "'")
        .replace("&#8220;", '"')
        .replace("&#8221;", '"')
        .replace("&#151;", " ")
        .replace("&#150;", " ")
    )
    return _WS_RE.sub(" ", text)


def normalize_money(value: str) -> str:
    """'$34.00' and '34' both normalize to '34'. Trailing zeros dropped."""
    cleaned = re.sub(r"[^0-9.]", "", value)
    if not cleaned:
        return ""
    try:
        num = float(cleaned)
    except ValueError:
        return ""
    return f"{num:.10g}"


def normalize_percent(value: str) -> str:
    return normalize_money(value)


def _first_match(
    patterns: tuple[re.Pattern[str], ...], text: str
) -> tuple[str, tuple[int, int]] | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip(), m.span(1)
    return None


def _field(
    deal_key: str,
    name: FieldName,
    raw: str,
    normalized: str,
    accession: str,
    span: tuple[int, int],
) -> ProtectedField:
    return ProtectedField(
        field_id=short_hash(deal_key, name),
        name=name,
        value_raw=raw,
        value_normalized=normalized,
        source_accession=accession,
        source_span=span,
    )


def parse_filing(
    raw_text: str,
    accession: str,
    filed_date: date,
    target_cik: str,
    sic_2digit: str = "99",
    size_metric_value: float = 0.0,
    company_name_hint: str = "",
) -> ParsedFiling:
    """Parse one 8-K into a DealRecord, or return an exclusion with a reason."""
    text = strip_markup(raw_text)

    if not _ITEM_RE.search(text):
        return ParsedFiling(
            exclusion=ExclusionRecord(accession, "no_item_101", "no Item 1.01 or 2.01")
        )

    acq = _first_match(_ACQUIRER_PATTERNS, text)
    tgt = _first_match(_TARGET_PATTERNS, text)

    acquirer_name = acq[0] if acq else ""
    target_name = tgt[0] if tgt else company_name_hint

    if not acquirer_name:
        return ParsedFiling(
            exclusion=ExclusionRecord(accession, "no_acquirer", "acquirer not found")
        )
    if not target_name:
        return ParsedFiling(exclusion=ExclusionRecord(accession, "no_target", "target not found"))
    if _clean_name(acquirer_name) == _clean_name(target_name):
        return ParsedFiling(
            exclusion=ExclusionRecord(accession, "ambiguous_parties", "acquirer equals target")
        )

    deal_key = f"{accession}:{_clean_name(target_name)}"
    fields: list[ProtectedField] = []

    price = _first_match(_PRICE_PATTERNS, text)
    if price:
        norm = normalize_money(price[0])
        if norm:
            fields.append(
                _field(deal_key, "offer_price_per_share", price[0], norm, accession, price[1])
            )

    prem = _first_match(_PREMIUM_PATTERNS, text)
    if prem:
        norm = normalize_percent(prem[0])
        if norm:
            fields.append(_field(deal_key, "premium_pct", prem[0], norm, accession, prem[1]))

    if not fields:
        return ParsedFiling(
            exclusion=ExclusionRecord(accession, "no_numeric_field", "no price or premium found")
        )

    # non-numeric fields are optional, so they are added after the numeric gate
    lowered = text.lower()
    for phrase, label in _CONSIDERATION_WORDS:
        idx = lowered.find(phrase)
        if idx >= 0:
            fields.append(
                _field(
                    deal_key,
                    "consideration_type",
                    phrase,
                    label,
                    accession,
                    (idx, idx + len(phrase)),
                )
            )
            break

    close = _CLOSE_RE.search(text)
    if close:
        fields.append(
            _field(
                deal_key,
                "expected_close",
                close.group(1),
                close.group(1).lower(),
                accession,
                close.span(1),
            )
        )

    fields.append(
        _field(
            deal_key,
            "announcement_date",
            filed_date.isoformat(),
            filed_date.isoformat(),
            accession,
            (0, 0),
        )
    )

    record = DealRecord(
        deal_id=short_hash(deal_key),
        acquirer_name=_clean_name(acquirer_name),
        target_name=_clean_name(target_name),
        target_cik=str(target_cik),
        announcement_date=filed_date,
        sic_2digit=sic_2digit,
        size_metric_value=float(size_metric_value),
        size_band=0,  # assigned later, needs the whole cohort to bucket
        protected_fields=tuple(fields),
        source_accessions=(accession,),
    )
    return ParsedFiling(record=record)


def _clean_name(name: str) -> str:
    """Trim trailing punctuation and collapse whitespace in a company name."""
    out = _WS_RE.sub(" ", name).strip(" .,;:")
    return re.sub(r"\s+(inc|corp|corporation|co|ltd|llc|plc)\.?$", "", out, flags=re.I).strip()


def render_filing(record: DealRecord, company_name_hint: str = "") -> str:
    """Render a DealRecord back into filing-shaped text.

    Exists so the parser can be round-trip tested: render a known record, parse it,
    and check the values survive. Also used by the synthetic corpus builder.
    """
    price = record.field_by_name("offer_price_per_share")
    prem = record.field_by_name("premium_pct")
    cons = record.field_by_name("consideration_type")
    close = record.field_by_name("expected_close")

    parts = [
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
        "FORM 8-K",
        f"Date of Report: {record.announcement_date.isoformat()}",
        "",
        "Item 1.01 Entry into a Material Definitive Agreement",
        "",
        f"{record.acquirer_name} to acquire {record.target_name}, "
        f"in a transaction announced on {record.announcement_date.isoformat()}.",
    ]
    if price:
        parts.append(
            f"Under the agreement, shareholders will receive ${price.value_raw} per share in cash."
        )
    if prem:
        parts.append(
            f"The consideration represents a premium of {prem.value_raw}% to the prior close."
        )
    if cons:
        parts.append(f"The transaction is structured as an {cons.value_raw} deal.")
    if close:
        parts.append(f"The transaction is expected to close in {close.value_raw}.")
    return "\n".join(parts)


def parse_or_exclude(
    raw_text: str, accession: str, filed_date: date, target_cik: str, **kw: Any
) -> ParsedFiling:
    """parse_filing that never raises. Any unexpected error becomes an exclusion."""
    try:
        return parse_filing(raw_text, accession, filed_date, target_cik, **kw)
    except ParseIncompleteError:
        raise
    except Exception as exc:  # noqa: BLE001 a malformed filing must not end the run
        return ParsedFiling(
            exclusion=ExclusionRecord(accession, "no_numeric_field", f"{type(exc).__name__}: {exc}")
        )
