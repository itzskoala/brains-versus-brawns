import pandas as pd
import pytest

from src.data.scraping import merge, pipeline
from src.data.scraping.ufcstats_browser import UfcStatsFetchError
from src.features.engineering import FIGHT_STAT_COLUMNS

COMPLETED_EVENTS_URL = pipeline.COMPLETED_EVENTS_URL


class FakeBrowser:
    """Stands in for UfcStatsBrowser in tests - no Playwright, no network.
    `pages` maps URL -> HTML string, or URL -> a list of responses popped
    one per call (an item that's an Exception instance is raised instead
    of returned, for simulating a flaky fetch that succeeds on a later
    attempt). Requesting a URL with nothing left to give is a test bug,
    not a silently-wrong result, so it raises loudly."""

    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, wait_ms=1500):
        self.requested.append(url)
        if url not in self.pages:
            raise AssertionError(f"test fixture has no canned page for {url}")
        value = self.pages[url]
        if isinstance(value, list):
            if not value:
                raise AssertionError(f"test fixture ran out of canned responses for {url}")
            item = value.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return value


def _write_minimal_existing_data(data_dir):
    totals = {col: 0 for pair in FIGHT_STAT_COLUMNS.values() for col in pair}
    pd.DataFrame([{"event_id": "e0", "event_name": "UFC Fight Night: Old",
                    "date": "2026-08-01", "location": "Las Vegas, Nevada, USA"}]
                 ).to_csv(data_dir / "event.csv", index=False)
    pd.DataFrame([{"event_id": "e0", "fight_id": "f0", "weight_class": "Lightweight",
                    "title_fight": False, "r_id": "r0", "b_id": "b0", "winner_id": "r0",
                    "result_status": "win", "method": "KO/TKO", "round": 1, "time": "1:00",
                    "time_format": "3 Rnd (5-5-5)", "referee": "Ref", "details": "Punch"}]
                 ).to_csv(data_dir / "fight.csv", index=False)
    pd.DataFrame([{"fight_id": "f0", "event_id": "e0", "event_name": "UFC Fight Night: Old",
                    "event_date": "2026-08-01", "event_location": "Las Vegas, Nevada, USA",
                    "weight_class": "Lightweight", "title_fight": False, "winner_id": "r0",
                    "result_status": "win", "method": "KO/TKO", "finish_round": 1, "finish_time": "1:00",
                    "time_format": "3 Rnd (5-5-5)", "referee": "Ref", "details": "Punch",
                    "bonuses": None, "r_fighter_id": "r0", "b_fighter_id": "b0", "rounds_fought": 1,
                    **totals}]).to_csv(data_dir / "fights.csv", index=False)
    # Real fight_rounds.csv is never empty (~25k rows) - an empty DataFrame
    # trips up clean_rounds' "mm:ss" split (no column to index into), which
    # is a non-issue in production but would be a self-inflicted bug in
    # these fixtures, so f0 gets one real round row.
    pd.DataFrame([{
        "fight_id": "f0", "round_no": 1, "r_id": "r0", "b_id": "b0", "r_kd": 0, "b_kd": 0,
        "r_sig_landed": 1, "r_sig_atmp": 1, "b_sig_landed": 0, "b_sig_atmp": 1,
        "r_total_str_landed": 1, "r_total_str_atmp": 1, "b_total_str_landed": 0, "b_total_str_atmp": 1,
        "r_td_success": 0, "r_td_atmp": 0, "b_td_success": 0, "b_td_atmp": 0,
        "r_sub_att": 0, "b_sub_att": 0, "r_rev": 0, "b_rev": 0, "r_ctrl": "0:10", "b_ctrl": "0:00",
        **{f"{s}_sig_str_{k}_{t}": 0 for s in "rb" for k in ("landed", "atmp")
           for t in ("head", "body", "leg", "distance", "clinch", "ground")},
    }]).to_csv(data_dir / "fight_rounds.csv", index=False)
    pd.DataFrame([
        {"fighter_id": "r0", "fighter_name": "Old Red", "fighter_nick_name": "", "height": "5' 10\"",
         "weight_lbs": 155, "reach_inches": 70, "stance": "Orthodox", "dob": "1990-01-01",
         "slpm": 3.0, "str_acc": 45, "sapm": 2.5, "str_def": 55, "td_avg": 1.0, "td_acc": 40,
         "td_def": 60, "sub_avg": 0.5},
        {"fighter_id": "b0", "fighter_name": "Old Blue", "fighter_nick_name": "", "height": "5' 11\"",
         "weight_lbs": 155, "reach_inches": 71, "stance": "Switch", "dob": "1991-01-01",
         "slpm": 2.8, "str_acc": 42, "sapm": 2.7, "str_def": 50, "td_avg": 0.8, "td_acc": 35,
         "td_def": 55, "sub_avg": 0.3},
    ]).to_csv(data_dir / "fighter.csv", index=False)
    pd.DataFrame(columns=["fight_id", "bonus_type"]).to_csv(data_dir / "fighter_bonus.csv", index=False)


@pytest.fixture
def data_dir(tmp_path):
    _write_minimal_existing_data(tmp_path)
    return tmp_path


def _completed_events_page(rows):
    """rows=[] renders a table with zero data rows - the shape UFCStats'
    own "ran past the last page" response takes, which parse_completed_
    events_list turns into a ParseError and discover_all_events reads as
    "pagination is done"."""
    trs = "\n".join(f"""
      <tr class="b-statistics__table-row">
        <td class="b-statistics__table-col"><i class="b-statistics__table-content">
          <a class="b-link b-link_style_black" href="http://ufcstats.com/event-details/{eid}">{name}</a>
          <span class="b-statistics__date">{date}</span></i></td>
        <td class="b-statistics__table-col">Las Vegas, Nevada, USA</td>
      </tr>""" for eid, name, date in rows)
    return f'<table class="b-statistics__table-events">{trs}</table>'


def _event_page(event_name, date, location, fight_id):
    return f"""
    <h2 class="b-content__title"><span class="b-content__title-highlight">{event_name}</span></h2>
    <ul class="b-list__box-list">
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Date:</i> {date}</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Location:</i> {location}</li>
    </ul>
    <table><tr data-link="http://ufcstats.com/fight-details/{fight_id}">
      <td><p>win</p></td>
      <td><a href="http://ufcstats.com/fighter-details/r1">Red Fighter</a>
          <a href="http://ufcstats.com/fighter-details/b1">Blue Fighter</a></td>
      <td>0<br>0</td><td>1<br>1</td><td>0<br>0</td><td>0<br>0</td>
      <td><p>Lightweight</p></td><td>KO/TKO</td><td>1</td><td>1:00</td>
    </tr></table>
    """


def _fight_page_html():
    stat_row = """
      <tr><td><p>Red Fighter</p><p>Blue Fighter</p></td>
        <td><p>0</p><p>0</p></td><td><p>1 of 1</p><p>0 of 1</p></td><td><p>100%</p><p>0%</p></td>
        <td><p>1 of 1</p><p>0 of 1</p></td><td><p>0 of 0</p><p>0 of 0</p></td><td><p>---</p><p>---</p></td>
        <td><p>0</p><p>0</p></td><td><p>0</p><p>0</p></td><td><p>0:10</p><p>0:00</p></td></tr>"""
    target_row = """
      <tr><td><p>Red Fighter</p><p>Blue Fighter</p></td>
        <td><p>1 of 1</p><p>0 of 1</p></td><td><p>100%</p><p>0%</p></td>
        <td><p>1 of 1</p><p>0 of 1</p></td><td><p>0 of 0</p><p>0 of 0</p></td>
        <td><p>0 of 0</p><p>0 of 0</p></td><td><p>1 of 1</p><p>0 of 1</p></td>
        <td><p>0 of 0</p><p>0 of 0</p></td><td><p>0 of 0</p><p>0 of 0</p></td></tr>"""
    return f"""
    <div class="b-fight-details__persons clearfix">
      <div class="b-fight-details__person"><i class="b-fight-details__person-status">W</i>
        <a class="b-fight-details__person-link" href="http://ufcstats.com/fighter-details/r1">Red Fighter</a></div>
      <div class="b-fight-details__person"><i class="b-fight-details__person-status">L</i>
        <a class="b-fight-details__person-link" href="http://ufcstats.com/fighter-details/b1">Blue Fighter</a></div>
    </div>
    <p><i class="b-fight-details__text-item_first"><i class="b-fight-details__label">Method:</i> KO/TKO</i>
       <i class="b-fight-details__text-item"><i class="b-fight-details__label">Round:</i> 1</i>
       <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time:</i> 1:00</i>
       <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time format:</i> 3 Rnd (5-5-5)</i>
       <i class="b-fight-details__text-item"><i class="b-fight-details__label">Referee:</i> <span>Ref</span></i></p>
    <p><i class="b-fight-details__label">Details:</i> Punch</p>
    <table><tr><th>h</th></tr>{stat_row}</table>
    <table><tr><th>h</th></tr>{target_row}</table>
    """


def _fighter_page_html(name):
    return f"""
    <h2 class="b-content__title"><span class="b-content__title-highlight">{name}</span></h2>
    <ul class="b-list__box-list">
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Height:</i> 5' 11"</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Weight:</i> 155 lbs.</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Reach:</i> 71"</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">STANCE:</i> Orthodox</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">DOB:</i> Jan 01, 1995</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">SLpM:</i> 3.0</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Str. Acc.:</i> 45%</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">SApM:</i> 2.0</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Str. Def:</i> 55%</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Avg.:</i> 1.0</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Acc.:</i> 40%</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Def.:</i> 60%</li>
      <li class="b-list__box-list-item"><i class="b-list__box-item-title">Sub. Avg.:</i> 0.0</li>
    </ul>
    """


EMPTY_PAGE = _completed_events_page([])


# ---------------------------------------------------------------------------
# discover_all_events: exhaustive pagination, no rolling-window cutoff
# ---------------------------------------------------------------------------

def test_discover_all_events_paginates_across_multiple_pages():
    page1 = _completed_events_page([("recent", "UFC Fight Night: Recent", "October 10, 2026")])
    page2 = _completed_events_page([("older", "UFC Fight Night: Older", "September 01, 2026")])
    page3 = _completed_events_page([("oldest", "UFC 1", "November 12, 1993")])
    browser = FakeBrowser({
        COMPLETED_EVENTS_URL: page1,
        f"{COMPLETED_EVENTS_URL}?page=2": page2,
        f"{COMPLETED_EVENTS_URL}?page=3": page3,
        f"{COMPLETED_EVENTS_URL}?page=4": EMPTY_PAGE,
    })
    events = pipeline.discover_all_events(browser)
    assert set(events) == {"recent", "older", "oldest"}
    assert browser.requested == [COMPLETED_EVENTS_URL, f"{COMPLETED_EVENTS_URL}?page=2",
                                  f"{COMPLETED_EVENTS_URL}?page=3", f"{COMPLETED_EVENTS_URL}?page=4"]


def test_discover_all_events_does_not_stop_at_an_old_date():
    # The whole point of exhaustive discovery: a much-older event on a
    # later page must still be collected, unlike a "stop once we're past
    # some cutoff" heuristic would.
    page1 = _completed_events_page([("recent", "UFC Fight Night: Recent", "October 10, 2026")])
    page2 = _completed_events_page([("ancient", "UFC 1", "November 12, 1993")])
    browser = FakeBrowser({
        COMPLETED_EVENTS_URL: page1,
        f"{COMPLETED_EVENTS_URL}?page=2": page2,
        f"{COMPLETED_EVENTS_URL}?page=3": EMPTY_PAGE,
    })
    events = pipeline.discover_all_events(browser)
    assert "ancient" in events


def test_discover_all_events_raises_if_pagination_never_terminates():
    # Every page looks non-empty forever - the "ran out" stop condition
    # never fires. That's a bug in the stop condition (or a markup
    # change), not a real 800-event-deep catalog, so this must raise
    # loudly rather than silently truncating or looping forever.
    browser = FakeBrowser({
        COMPLETED_EVENTS_URL: _completed_events_page([("e1", "Show", "October 10, 2026")]),
        **{f"{COMPLETED_EVENTS_URL}?page={n}": _completed_events_page([(f"e{n}", "Show", "October 10, 2026")])
           for n in range(2, 10)},
    })
    with pytest.raises(pipeline.PipelineError):
        pipeline.discover_all_events(browser, max_pages=5)


def test_discover_all_events_retries_transient_fetch_failures():
    browser = FakeBrowser({
        COMPLETED_EVENTS_URL: [UfcStatsFetchError("timeout"),
                                _completed_events_page([("e1", "Show", "October 10, 2026")])],
        f"{COMPLETED_EVENTS_URL}?page=2": EMPTY_PAGE,
    })
    events = pipeline.discover_all_events(browser)
    assert set(events) == {"e1"}


def test_discover_all_events_gives_up_after_exhausting_retries():
    browser = FakeBrowser({
        COMPLETED_EVENTS_URL: [UfcStatsFetchError("down"), UfcStatsFetchError("down"),
                                UfcStatsFetchError("still down")],
    })
    with pytest.raises(UfcStatsFetchError):
        pipeline.discover_all_events(browser)


# ---------------------------------------------------------------------------
# find_missing_events: diff by event_id, not name/date
# ---------------------------------------------------------------------------

def test_find_missing_events_diffs_by_event_id(data_dir):
    discovered = {
        "e0": {"event_id": "e0", "event_name": "Totally different headline for the same show",
               "event_date": pd.Timestamp("2026-08-01").date(), "location": "Las Vegas, Nevada, USA"},
        "new1": {"event_id": "new1", "event_name": "New Show",
                 "event_date": pd.Timestamp("2026-09-01").date(), "location": "Las Vegas, Nevada, USA"},
    }
    missing = pipeline.find_missing_events(discovered, str(data_dir))
    assert [e["event_id"] for e in missing] == ["new1"]


def test_find_missing_events_sorted_oldest_first(data_dir):
    discovered = {
        "newer": {"event_id": "newer", "event_name": "Newer",
                  "event_date": pd.Timestamp("2026-09-08").date(), "location": "x"},
        "older": {"event_id": "older", "event_name": "Older",
                  "event_date": pd.Timestamp("2026-09-01").date(), "location": "x"},
    }
    missing = pipeline.find_missing_events(discovered, str(data_dir))
    assert [e["event_id"] for e in missing] == ["older", "newer"]


# ---------------------------------------------------------------------------
# run(): per-event atomic processing, duplicate prevention, idempotency
# ---------------------------------------------------------------------------

def _full_pages(event_id="new1", name="UFC Fight Night: New Show", date="September 01, 2026",
                 fight_id="fnew1"):
    event_url = f"http://ufcstats.com/event-details/{event_id}"
    fight_url = f"http://ufcstats.com/fight-details/{fight_id}"
    return {
        COMPLETED_EVENTS_URL: _completed_events_page([(event_id, name, date)]),
        f"{COMPLETED_EVENTS_URL}?page=2": EMPTY_PAGE,
        event_url: _event_page(name, date, "Las Vegas, Nevada, USA", fight_id),
        fight_url: _fight_page_html(),
        "http://ufcstats.com/fighter-details/r1": _fighter_page_html("Red Fighter"),
        "http://ufcstats.com/fighter-details/b1": _fighter_page_html("Blue Fighter"),
    }


def test_run_writes_new_event(data_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(_full_pages()))

    result = pipeline.run(data_dir=str(data_dir), dry_run=False)

    assert result["status"] == "written"
    assert result["added"] == 1
    assert result["failed"] == 0
    fights = pd.read_csv(data_dir / "fights.csv")
    assert "fnew1" in set(fights["fight_id"])
    events = pd.read_csv(data_dir / "event.csv")
    assert "new1" in set(events["event_id"])
    fighters = pd.read_csv(data_dir / "fighter.csv")
    assert {"r1", "b1"}.issubset(set(fighters["fighter_id"]))


def test_run_dry_run_does_not_write(data_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(_full_pages()))

    before = (data_dir / "fights.csv").read_text()
    result = pipeline.run(data_dir=str(data_dir), dry_run=True)

    assert result["status"] == "dry_run"
    assert (data_dir / "fights.csv").read_text() == before


def test_run_no_new_events_when_nothing_missing(data_dir, monkeypatch):
    pages = {
        COMPLETED_EVENTS_URL: _completed_events_page([("e0", "UFC Fight Night: Old", "August 01, 2026")]),
        f"{COMPLETED_EVENTS_URL}?page=2": EMPTY_PAGE,
    }
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(pages))

    result = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert result["status"] == "no_new_events"
    assert result["missing"] == 0


def test_run_rediscovering_an_event_already_written_adds_nothing(data_dir, monkeypatch):
    # Idempotency/duplicate-prevention: running twice against the same
    # discovered catalog must not duplicate rows the second time.
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(_full_pages()))
    first = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert first["added"] == 1

    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(_full_pages()))
    second = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert second["status"] == "no_new_events"

    fights = pd.read_csv(data_dir / "fights.csv")
    assert list(fights["fight_id"]).count("fnew1") == 1


def test_run_one_bad_event_does_not_block_or_undo_a_good_one(data_dir, monkeypatch):
    # The core resilience property: events are processed and persisted one
    # at a time. A malformed fight page on event B must not roll back
    # event A, which already scraped and validated cleanly.
    good = _full_pages(event_id="good1", name="UFC Fight Night: Good", date="August 15, 2026",
                        fight_id="fgood1")
    bad = _full_pages(event_id="bad1", name="UFC Fight Night: Bad", date="August 22, 2026",
                       fight_id="fbad1")
    pages = {**good, **bad}
    pages[COMPLETED_EVENTS_URL] = _completed_events_page([
        ("good1", "UFC Fight Night: Good", "August 15, 2026"),
        ("bad1", "UFC Fight Night: Bad", "August 22, 2026"),
    ])
    pages[f"{COMPLETED_EVENTS_URL}?page=2"] = EMPTY_PAGE
    pages["http://ufcstats.com/fight-details/fbad1"] = "<html><body>not a real fight page</body></html>"
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(pages))

    result = pipeline.run(data_dir=str(data_dir), dry_run=False)

    assert result["added"] == 1
    assert result["failed"] == 1
    assert result["failed_events"][0]["event_id"] == "bad1"

    events = pd.read_csv(data_dir / "event.csv")
    fights = pd.read_csv(data_dir / "fights.csv")
    assert "good1" in set(events["event_id"])
    assert "fgood1" in set(fights["fight_id"])
    # the bad event must be entirely absent - never partially written
    assert "bad1" not in set(events["event_id"])
    assert "fbad1" not in set(fights["fight_id"])


def test_run_incomplete_event_is_retried_next_run_not_marked_done(data_dir, monkeypatch):
    # An event that fails this run must still show up as "missing" next
    # run (no event.csv row, no half-written fight rows to collide with).
    bad_pages = _full_pages(event_id="bad1", name="UFC Fight Night: Bad", date="August 22, 2026",
                             fight_id="fbad1")
    bad_pages["http://ufcstats.com/fight-details/fbad1"] = "<html><body>broken</body></html>"
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(bad_pages))
    first = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert first["failed"] == 1
    assert first["added"] == 0

    # Next run: same event still comes back as discovered+missing, and
    # this time it scrapes cleanly.
    good_pages = _full_pages(event_id="bad1", name="UFC Fight Night: Bad", date="August 22, 2026",
                              fight_id="fbad1")
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(good_pages))
    second = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert second["added"] == 1
    assert second["failed"] == 0

    events = pd.read_csv(data_dir / "event.csv")
    fights = pd.read_csv(data_dir / "fights.csv")
    assert "bad1" in set(events["event_id"])
    assert "fbad1" in set(fights["fight_id"])


def test_run_retries_a_transient_fetch_failure_within_the_run(data_dir, monkeypatch):
    pages = _full_pages()
    event_url = "http://ufcstats.com/event-details/new1"
    pages[event_url] = [UfcStatsFetchError("timeout"), pages[event_url]]
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(pages))

    result = pipeline.run(data_dir=str(data_dir), dry_run=False)
    assert result["added"] == 1
    assert result["failed"] == 0


def test_run_limit_caps_events_processed_this_run(data_dir, monkeypatch):
    pages = {
        **_full_pages(event_id="a1", name="UFC Fight Night: A", date="August 01, 2026", fight_id="fa1"),
        **_full_pages(event_id="b1", name="UFC Fight Night: B", date="August 08, 2026", fight_id="fb1"),
    }
    pages[COMPLETED_EVENTS_URL] = _completed_events_page([
        ("a1", "UFC Fight Night: A", "August 01, 2026"),
        ("b1", "UFC Fight Night: B", "August 08, 2026"),
    ])
    pages[f"{COMPLETED_EVENTS_URL}?page=2"] = EMPTY_PAGE
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(pages))

    result = pipeline.run(data_dir=str(data_dir), dry_run=False, limit=1)
    assert result["added"] == 1
    events = pd.read_csv(data_dir / "event.csv")
    assert "a1" in set(events["event_id"])
    assert "b1" not in set(events["event_id"])


def test_run_aborts_whole_event_on_validation_failure_not_other_events(data_dir, monkeypatch):
    pages = _full_pages()
    monkeypatch.setattr(pipeline, "UfcStatsBrowser", lambda: FakeBrowser(pages))
    monkeypatch.setattr(merge, "validate_new_data",
                         lambda updated: (_ for _ in ()).throw(merge.ValidationError("forced failure")))

    before = (data_dir / "fights.csv").read_text()
    result = pipeline.run(data_dir=str(data_dir), dry_run=False)

    assert result["added"] == 0
    assert result["failed"] == 1
    assert (data_dir / "fights.csv").read_text() == before
