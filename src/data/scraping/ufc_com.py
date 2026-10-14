"""Detection-only source: ufc.com/events' own "Past Events" listing tells us
an event has concluded (https://www.ufc.com/events, #events-list-past). No
stats live here - ufc.com's cards are just result summaries. Every actual
stat comes from UFCStats.com (src.data.scraping.ufcstats_parse); this module
only answers "which events do we not have yet".
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

EVENTS_URL = "https://www.ufc.com/events"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# UFC event cards embed the start time as a Unix timestamp, but the date UFC
# (and UFCStats, and this dataset) label the card with is the US Eastern
# calendar date of that start time - converting the timestamp to UTC first
# can land on the next day (e.g. an 8pm EDT Saturday start is already
# Sunday 00:00 UTC) and silently shift every event one day late.
CARD_TIMEZONE = ZoneInfo("America/New_York")


class UfcComParseError(Exception):
    """ufc.com/events didn't have the structure this parser expects."""


def fetch_events_html():
    resp = requests.get(EVENTS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_past_events(html):
    """Returns the "Past Events" cards from an already-fetched ufc.com/events
    page: [{"name": str, "date": datetime.date}, ...], oldest/newest order as
    the page lists them (most recent first). Raises UfcComParseError if the
    page doesn't look like ufc.com/events at all (layout change, error page,
    etc.) - callers should treat that as "stop, don't guess", not "zero new
    events"."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.find(id="events-list-past")
    if container is None:
        raise UfcComParseError(
            "ufc.com/events: #events-list-past not found - page structure may have changed")

    events = []
    for card in container.select("article.c-card-event--result"):
        headline = card.select_one(".c-card-event--result__headline a")
        date_div = card.select_one(".c-card-event--result__date")
        timestamp = date_div.get("data-main-card-timestamp") if date_div else None
        if headline is None or not timestamp:
            continue
        name = " ".join(headline.get_text(" ", strip=True).split())
        date = datetime.fromtimestamp(int(timestamp), tz=CARD_TIMEZONE).date()
        events.append({"name": name, "date": date})

    if not events:
        raise UfcComParseError(
            "ufc.com/events: #events-list-past found but zero event cards parsed - markup likely changed")
    return events


def get_past_events():
    return parse_past_events(fetch_events_html())
