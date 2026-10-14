import pytest

from src.data.scraping.ufc_com import UfcComParseError, parse_past_events

# Trimmed to the real structure confirmed against a live fetch of
# https://www.ufc.com/events on 2026-10-14: two <details> sections keyed by
# id ("events-list-upcoming" / "events-list-past"), each card an
# <article class="c-card-event--result">. Upcoming-card timestamps are the
# real event start time; past-card timestamps were found NOT to be (one
# real past card's timestamp decoded to 5am EDT, "Road to UFC" developmental
# programming, not the UFC-numbered show listed right next to it on the
# same page) - so these tests only rely on #events-list-past being present
# at all and the date/name actually on the card, not on the timestamp being
# semantically the event start time.
PAGE_HTML = """
<html><body>
<details id="events-list-upcoming">
  <article class="c-card-event--result">
    <h3 class="c-card-event--result__headline"><a href="/event/x">Buckley vs Malott</a></h3>
    <div class="c-card-event--result__date" data-main-card-timestamp="1792281600"></div>
  </article>
</details>
<details id="events-list-past">
  <article class="c-card-event--result">
    <h3 class="c-card-event--result__headline"><a href="/event/y">Allen vs Duncan</a></h3>
    <div class="c-card-event--result__date" data-main-card-timestamp="1760054400"></div>
  </article>
  <article class="c-card-event--result">
    <h3 class="c-card-event--result__headline"><a href="/event/z">Silva vs Wang</a></h3>
    <div class="c-card-event--result__date" data-main-card-timestamp="1759449600"></div>
  </article>
</details>
</body></html>
"""


def test_parses_only_past_events_not_upcoming():
    events = parse_past_events(PAGE_HTML)
    names = [e["name"] for e in events]
    assert names == ["Allen vs Duncan", "Silva vs Wang"]


def test_each_event_has_a_date():
    events = parse_past_events(PAGE_HTML)
    assert all(e["date"] is not None for e in events)


def test_missing_past_events_container_raises():
    html = "<html><body><details id='events-list-upcoming'></details></body></html>"
    with pytest.raises(UfcComParseError):
        parse_past_events(html)


def test_past_events_container_with_zero_cards_raises():
    html = "<html><body><details id='events-list-past'></details></body></html>"
    with pytest.raises(UfcComParseError):
        parse_past_events(html)


def test_malformed_html_missing_date_div_is_skipped_not_crashed():
    html = """
    <details id="events-list-past">
      <article class="c-card-event--result">
        <h3 class="c-card-event--result__headline"><a href="/event/y">No Date Card</a></h3>
      </article>
      <article class="c-card-event--result">
        <h3 class="c-card-event--result__headline"><a href="/event/z">Has Date</a></h3>
        <div class="c-card-event--result__date" data-main-card-timestamp="1760054400"></div>
      </article>
    </details>
    """
    events = parse_past_events(html)
    assert [e["name"] for e in events] == ["Has Date"]
