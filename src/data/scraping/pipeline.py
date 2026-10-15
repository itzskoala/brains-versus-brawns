"""Orchestrates the auto-update: UFCStats.com's own completed-events list
is both the discovery source and the stats source for every new event -
there is no second feed (e.g. ufc.com's "Past Events" widget, which only
shows a rolling recent window and silently drops anything that scrolls
off it) that can drift out of sync or quietly skip history. Discovery
paginates that list end-to-end every run, so an event missed months ago
is exactly as discoverable as one from last week.

Each missing event is scraped, validated, and written to data/*.csv one at
a time - not batched across the whole run. That means one bad event
(changed markup, a fight with a shape we don't handle, a transient
network blip) only costs that event: everything already written stays
written, and the bad event is simply still "missing" next run, so it gets
retried automatically with no extra bookkeeping. An event only ever
appears in data/event.csv once every one of its fights is already durably
on disk (see merge.FILES) - so "in event.csv" always means "fully
ingested," never "ingestion started."

Usage: python -m src.data.scraping.pipeline [--dry-run] [--data-dir data] [--limit N]
"""
import argparse
import sys

import pandas as pd

from src.data.scraping import merge
from src.data.scraping import ufcstats_parse as parse
from src.data.scraping.ufcstats_browser import UfcStatsBrowser, UfcStatsFetchError

COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed"

# UFCStats has ~800 completed events / 25 per page as of 2026. A page
# count anywhere near this means the "page came back empty" stop
# condition broke (e.g. a markup change makes every page look non-empty)
# - not that the UFC has suddenly run 5000+ events - so treat hitting it
# as a bug to surface, not a number to raise.
MAX_PAGES = 200

# A single flaky fetch (timeout, dropped connection) shouldn't cost an
# event a full week's retry cycle - but a page that fetched fine and still
# doesn't parse (ParseError/ChallengePageError) won't parse differently on
# attempt 2, so those aren't retried here at all.
FETCH_RETRIES = 3


class PipelineError(Exception):
    """Discovery itself broke (pagination never terminated) - this is
    always a bug to fix, never a signal to silently narrow the scan."""


def discover_all_events(browser, max_pages=MAX_PAGES):
    """Paginates http://ufcstats.com/statistics/events/completed (and its
    ?page=N continuations) to exhaustion. Returns every event UFCStats has
    ever marked completed as {event_id: {"event_id", "event_name",
    "event_date", "location"}} - the whole history, not a recent window.
    The only stop condition is a page with zero rows (the list ran out);
    there is no date-based early exit, because the entire point is to
    never again assume "anything older than X is already accounted for."
    """
    events = {}
    for page_no in range(1, max_pages + 1):
        url = COMPLETED_EVENTS_URL if page_no == 1 else f"{COMPLETED_EVENTS_URL}?page={page_no}"
        html = _fetch_with_retries(browser, url)
        try:
            page_events = parse.parse_completed_events_list(html, url)
        except parse.ParseError:
            break  # past the last page - UFCStats renders an empty table here
        for event in page_events:
            events[event["event_id"]] = event
    else:
        raise PipelineError(
            f"pagination hit the {max_pages}-page safety ceiling without running out of "
            "pages - the 'empty page' stop condition likely broke (markup change?), this "
            "is not evidence the UFC ran that many events")
    return events


def find_missing_events(discovered_events, data_dir="data"):
    """discovered_events (from discover_all_events) vs. data/event.csv, by
    event_id - UFCStats' own id is the join key now that it is both the
    discovery and the stats source, so no cross-source name/date fuzzing
    is needed. Returns the events we don't have yet, oldest first."""
    existing_ids = set(pd.read_csv(f"{data_dir}/event.csv")["event_id"])
    missing = [e for e in discovered_events.values() if e["event_id"] not in existing_ids]
    return sorted(missing, key=lambda e: e["event_date"])


def _fetch_with_retries(browser, url, attempts=FETCH_RETRIES):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return browser.get(url)
        except UfcStatsFetchError as exc:
            last_exc = exc
            print(f"[pipeline] fetch failed for {url} (attempt {attempt}/{attempts}): {exc}")
    raise last_exc


def scrape_event(browser, event_id):
    """One UFCStats event, fully: event meta, every fight (totals +
    round-by-round), every fighter who appeared. Returns (event_meta,
    fight_bundles, fighter_rows) - see merge.build_fight_bundle for the
    bundle shape. Raises on the first bad page (ParseError) or exhausted
    retries (UfcStatsFetchError) - there is no partial result; the caller
    either gets a fully-scraped event or nothing."""
    event_url = f"http://ufcstats.com/event-details/{event_id}"
    event_page = parse.parse_event_page(_fetch_with_retries(browser, event_url), event_url)
    event_meta = {
        "event_id": event_id,
        "event_name": event_page["event_name"],
        "event_date": event_page["event_date"],
        "location": event_page["location"],
    }

    bundles, fighter_ids = [], set()
    for event_fight in event_page["fights"]:
        fight_url = f"http://ufcstats.com/fight-details/{event_fight['fight_id']}"
        fight_detail = parse.parse_fight_page(
            _fetch_with_retries(browser, fight_url), event_fight["fight_id"], fight_url)
        bundles.append(merge.build_fight_bundle(event_meta, event_fight, fight_detail))
        fighter_ids.update([fight_detail["r_fighter_id"], fight_detail["b_fighter_id"]])

    fighter_rows = []
    for fighter_id in sorted(fighter_ids):
        fighter_url = f"http://ufcstats.com/fighter-details/{fighter_id}"
        fighter_rows.append(parse.parse_fighter_page(
            _fetch_with_retries(browser, fighter_url), fighter_id, fighter_url))

    return event_meta, bundles, fighter_rows


def _apply_event(updated, event_meta, bundles, fighter_rows):
    """One event's rows merged into `updated` (the dict load_existing/
    write_all pass around) - a new dict, so a failed validate_new_data on
    the result never mutates the caller's already-good state. Mirrors
    merge.build_fight_bundle's per-file row shapes, just for one event
    instead of a whole run's worth."""
    updated = dict(updated)
    event_csv_row = {"event_id": event_meta["event_id"], "event_name": event_meta["event_name"],
                      "date": event_meta["event_date"], "location": event_meta["location"]}
    updated["event.csv"], _, _ = merge.append_new(updated["event.csv"], [event_csv_row], "event_id")
    updated["fight.csv"], _, _ = merge.append_new(
        updated["fight.csv"], [b["fight_row"] for b in bundles], "fight_id")
    updated["fights.csv"], _, _ = merge.append_new(
        updated["fights.csv"], [b["fights_row"] for b in bundles], "fight_id")
    round_rows = [row for b in bundles for row in b["fight_rounds_rows"]]
    updated["fight_rounds.csv"], _, _ = merge.append_new(
        updated["fight_rounds.csv"], round_rows, ["fight_id", "round_no"])
    bonus_rows = [row for b in bundles for row in b["bonus_rows"]]
    updated["fighter_bonus.csv"], _, _ = merge.append_new(
        updated["fighter_bonus.csv"], bonus_rows, ["fight_id", "bonus_type"])
    updated["fighter.csv"], _, _ = merge.upsert_fighters(updated["fighter.csv"], fighter_rows)
    return updated


def run(data_dir="data", dry_run=False, limit=None):
    with UfcStatsBrowser() as browser:
        discovered = discover_all_events(browser)
        missing = find_missing_events(discovered, data_dir)
        if limit is not None:
            missing = missing[:limit]

        print(f"[pipeline] {len(discovered)} event(s) discovered on UFCStats "
              f"(full history, not a recent window)")

        if not missing:
            print("[pipeline] every discovered event is already in data/event.csv - nothing to do")
            return {"status": "no_new_events", "discovered": len(discovered),
                    "missing": 0, "added": 0, "failed": 0, "failed_events": []}

        print(f"[pipeline] {len(missing)} missing from data/event.csv: "
              + ", ".join(f"{e['event_name']} ({e['event_date']})" for e in missing))

        updated = merge.load_existing(data_dir)
        added, failed = [], []
        for event in missing:
            label = f"{event['event_name']} ({event['event_date']})"
            try:
                event_meta, bundles, fighter_rows = scrape_event(browser, event["event_id"])
            except (parse.ParseError, UfcStatsFetchError) as exc:
                print(f"[pipeline] FAILED scraping '{label}': {exc} - "
                      f"nothing written for this event, will retry next run")
                failed.append({"event_id": event["event_id"], "event_name": event["event_name"],
                                "event_date": str(event["event_date"]), "error": str(exc)})
                continue

            candidate = _apply_event(updated, event_meta, bundles, fighter_rows)
            try:
                merge.validate_new_data(candidate)
            except merge.ValidationError as exc:
                print(f"[pipeline] FAILED validating '{label}': {exc} - "
                      f"nothing written for this event, will retry next run")
                failed.append({"event_id": event["event_id"], "event_name": event["event_name"],
                                "event_date": str(event["event_date"]), "error": str(exc)})
                continue

            updated = candidate
            if not dry_run:
                merge.write_all(data_dir, updated)
            added.append(event)
            print(f"[pipeline]   wrote '{label}': {len(bundles)} fight(s), "
                  f"{sum(len(b['fight_rounds_rows']) for b in bundles)} round row(s), "
                  f"{len(fighter_rows)} fighter page(s)")

    status = "dry_run" if dry_run else ("written" if added else "no_new_events")
    summary = {"status": status, "discovered": len(discovered), "missing": len(missing),
               "added": len(added), "failed": len(failed), "failed_events": failed}
    print(f"[pipeline] {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="scrape and validate, but don't write data/*.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None,
                         help="scrape at most N missing events this run (for testing/throttling)")
    args = parser.parse_args()

    try:
        result = run(data_dir=args.data_dir, dry_run=args.dry_run, limit=args.limit)
    except PipelineError as exc:
        print(f"[pipeline] FAILED: {exc}", file=sys.stderr)
        return 1

    return 0 if not result["failed_events"] else 1


if __name__ == "__main__":
    sys.exit(main())
