"""Natural-language date extraction from search queries (deterministic `now`)."""

from __future__ import annotations

import datetime as dt

from iris.query_parse import parse_dates

NOW = dt.datetime(2026, 8, 1, 12, 0, 0)


def _range(query: str) -> tuple[str, str | None, str | None]:
    p = parse_dates(query, now=NOW)
    fr = dt.date.fromtimestamp(p.date_from).isoformat() if p.date_from is not None else None
    to = dt.date.fromtimestamp(p.date_to).isoformat() if p.date_to is not None else None
    return p.cleaned_query, fr, to


def test_bare_year() -> None:
    assert _range("beach 2024") == ("beach", "2024-01-01", "2024-12-31")


def test_month_and_year_either_order() -> None:
    assert _range("trip july 2023") == ("trip", "2023-07-01", "2023-07-31")
    assert _range("2023 july trip") == ("trip", "2023-07-01", "2023-07-31")


def test_iso_forms() -> None:
    assert _range("receipts 2022-08") == ("receipts", "2022-08-01", "2022-08-31")
    assert _range("2020-05-17 party") == ("party", "2020-05-17", "2020-05-17")


def test_relative_phrases() -> None:
    assert _range("hike yesterday")[1:] == ("2026-07-31", "2026-07-31")
    q, fr, to = _range("photos from last month")
    assert q == "photos from" and fr == "2026-07-01"


def test_last_summer_is_previous_completed_season() -> None:
    # August 2026 is still summer 2026, so "last summer" = 2025.
    assert _range("sunset last summer") == ("sunset", "2025-06-01", "2025-08-31")


def test_no_date_leaves_query_untouched() -> None:
    p = parse_dates("a dog on a beach", now=NOW)
    assert p.cleaned_query == "a dog on a beach"
    assert p.date_from is None and p.label is None


def test_ambiguous_bare_month_word_is_ignored() -> None:
    # "may" as a verb must not become a date filter.
    p = parse_dates("i may go to the park", now=NOW)
    assert p.date_from is None
