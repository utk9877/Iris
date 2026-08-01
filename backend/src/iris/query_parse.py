"""Natural-language date extraction from a search query (ARCHITECTURE §4).

Pulls a date expression out of free text — "beach 2024", "trip july 2023", "last summer",
"photos from last month" — returning the leftover words (for CLIP/OCR ranking) plus a
``[date_from, date_to]`` epoch range. Pure stdlib, no dependency. Ranges are half-open in
spirit but returned inclusive (``date_to`` = last instant of the period) to match the
``sort_at >= from AND sort_at <= to`` filter. Ambiguous bare month names are deliberately
ignored (e.g. "may" the word), so only explicit or clearly-dated phrases trigger a filter.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {
    m.lower(): i for i, m in enumerate(calendar.month_abbr) if m
}

# Northern-hemisphere seasons -> (start_month, length_months).
_SEASONS = {
    "spring": (3, 3),
    "summer": (6, 3),
    "fall": (9, 3),
    "autumn": (9, 3),
    "winter": (12, 3),
}


@dataclass
class DateParse:
    cleaned_query: str
    date_from: float | None = None
    date_to: float | None = None
    label: str | None = None  # human summary of what was parsed, e.g. "2024" or "Jul 2023"


def _epoch(year: int, month: int, day: int) -> float:
    return dt.datetime(year, month, day).timestamp()


def _month_span(year: int, month: int) -> tuple[float, float]:
    last = calendar.monthrange(year, month)[1]
    return _epoch(year, month, 1), dt.datetime(year, month, last, 23, 59, 59).timestamp()


def _year_span(year: int) -> tuple[float, float]:
    return _epoch(year, 1, 1), dt.datetime(year, 12, 31, 23, 59, 59).timestamp()


def _day_span(d: dt.date) -> tuple[float, float]:
    start = dt.datetime(d.year, d.month, d.day)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp() - 1


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    index = (year * 12 + (month - 1)) + delta
    return index // 12, index % 12 + 1


def _season_span(year: int, season: str) -> tuple[float, float]:
    start_month, length = _SEASONS[season]
    end_year, end_month = _add_months(year, start_month, length - 1)
    return _month_span(year, start_month)[0], _month_span(end_year, end_month)[1]


def _clean(query: str, span: tuple[int, int]) -> str:
    text = query[: span[0]] + " " + query[span[1] :]
    return re.sub(r"\s+", " ", text).strip()


def parse_dates(query: str, *, now: dt.datetime | None = None) -> DateParse:
    """Extract the first date expression from ``query`` (see module docstring)."""
    now = now or dt.datetime.now()
    lower = query.lower()

    def result(m: re.Match[str], lo: float, hi: float, label: str) -> DateParse:
        return DateParse(_clean(query, m.span()), lo, hi, label)

    # 1. ISO day  YYYY-MM-DD
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lower):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        try:
            lo, hi = _day_span(dt.date(y, mo, d))
            return result(m, lo, hi, f"{y}-{mo:02d}-{d:02d}")
        except ValueError:
            pass
    # 2. ISO month  YYYY-MM
    if m := re.search(r"\b(\d{4})-(\d{1,2})\b", lower):
        y, mo = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            lo, hi = _month_span(y, mo)
            return result(m, lo, hi, f"{calendar.month_abbr[mo]} {y}")
    # 3. Month name + year (either order)
    names = "|".join(_MONTHS)
    if m := re.search(rf"\b({names})\.?\s+(\d{{4}})\b", lower) or re.search(
        rf"\b(\d{{4}})\s+({names})\.?\b", lower
    ):
        groups = m.groups()
        name = groups[0] if groups[0] in _MONTHS else groups[1]
        year = int(groups[1] if groups[0] in _MONTHS else groups[0])
        mo = _MONTHS[name]
        lo, hi = _month_span(year, mo)
        return result(m, lo, hi, f"{calendar.month_abbr[mo]} {year}")
    # 4. Season + year, or "last <season>"
    seasons = "|".join(_SEASONS)
    if m := re.search(rf"\b({seasons})\s+(\d{{4}})\b", lower):
        lo, hi = _season_span(int(m[2]), m[1])
        return result(m, lo, hi, f"{m[1].title()} {m[2]}")
    if m := re.search(rf"\blast\s+({seasons})\b", lower):
        season = m[1]
        year = now.year if now.month >= _SEASONS[season][0] + _SEASONS[season][1] else now.year - 1
        lo, hi = _season_span(year, season)
        return result(m, lo, hi, f"{season.title()} {year}")
    # 5. Relative phrases
    if m := re.search(r"\byesterday\b", lower):
        lo, hi = _day_span((now - dt.timedelta(days=1)).date())
        return result(m, lo, hi, "yesterday")
    if m := re.search(r"\btoday\b", lower):
        lo, hi = _day_span(now.date())
        return result(m, lo, hi, "today")
    if m := re.search(r"\b(this|last|past)\s+(day|week|month|year)s?\b", lower):
        lo, hi = _relative_span(m[1], 1, m[2], now)
        return result(m, lo, hi, f"{m[1]} {m[2]}")
    if m := re.search(r"\b(?:last|past)\s+(\d{1,3})\s+(day|week|month|year)s?\b", lower):
        lo, hi = _relative_span("last", int(m[1]), m[2], now)
        return result(m, lo, hi, f"last {m[1]} {m[2]}s")
    # 6. Bare year
    if m := re.search(r"\b(19\d{2}|20\d{2})\b", lower):
        y = int(m[1])
        lo, hi = _year_span(y)
        return result(m, lo, hi, str(y))

    return DateParse(query.strip())


def _relative_span(kind: str, n: int, unit: str, now: dt.datetime) -> tuple[float, float]:
    end = now.timestamp()
    if unit == "day":
        start = (now - dt.timedelta(days=n)).timestamp()
    elif unit == "week":
        start = (now - dt.timedelta(weeks=n)).timestamp()
    elif unit == "month":
        y, mo = _add_months(now.year, now.month, -n)
        start = _epoch(y, mo, min(now.day, calendar.monthrange(y, mo)[1]))
    else:  # year
        start = _epoch(now.year - n, now.month, min(now.day, 28))
    if kind == "this":  # "this month/year/week" -> the current calendar period to date
        if unit == "year":
            start = _epoch(now.year, 1, 1)
        elif unit == "month":
            start = _epoch(now.year, now.month, 1)
        elif unit == "week":
            start = (now - dt.timedelta(days=now.weekday())).timestamp()
    return start, end
