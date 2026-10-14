"""Orchestrates the auto-update: ufc.com/events tells us which events have
concluded, UFCStats.com supplies every stat for them, everything gets
validated against the exact cleaning path the rest of the project already
relies on, and only then is it written. This is the script
.github/workflows/update-fight-data.yml actually runs.

All-or-nothing by construction: every event this run finds is scraped and
held in memory first; nothing touches data/*.csv until every one of them
has parsed cleanly AND the combined dataset passes validate_new_data(). One
bad scrape (changed markup, a fight with a shape we don't handle) aborts
the whole run with nothing written and nothing retrained - never a partial
update.

Usage: python -m src.data.scraping.pipeline [--dry-run] [--data-dir data]
"""
import argparse
import difflib
import sys

import pandas as pd

from src.data.scraping import merge, ufc_com
from src.data.scraping import ufcstats_parse as parse
from src.data.scraping.ufcstats_browser import UfcStatsBrowser

COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed"
MAX_SEARCH_PAGES = 10


class PipelineError(Exception):
    """A missing event couldn't be scraped end-to-end - nothing is written."""


def find_missing_events(data_dir="data"):
    """ufc.com's "Past Events" vs. our own data/event.csv, by date (ufc.com
    and UFCStats format the same event's name differently - "Buckley vs
    Malott" vs. "UFC Fight Night: Buckley vs. Malott" - so date, not name,
    is the reliable join key here; a UFC show date is effectively unique).
    Returns ufc.com's event dicts we don't have yet, oldest first."""
    existing_dates = set(parse_existing_dates(data_dir))
    ufc_events = ufc_com.get_past_events()
    missing = [e for e in ufc_events if e["date"] not in existing_dates]
    return sorted(missing, key=lambda e: e["date"])


def parse_existing_dates(data_dir):
    df = pd.read_csv(f"{data_dir}/event.csv")
    return set(pd.to_datetime(df["date"]).dt.date)


def find_ufcstats_event(browser, name, date, max_pages=MAX_SEARCH_PAGES):
    """Paginates UFCStats' completed-events list looking for an event on
    `date`. Returns its {"event_id", "event_name", ...} dict, or None if
    UFCStats hasn't posted it yet (e.g. results take a day to go up) -
    callers should skip, not fail, on None."""
    for page_no in range(1, max_pages + 1):
        url = COMPLETED_EVENTS_URL if page_no == 1 else f"{COMPLETED_EVENTS_URL}?page={page_no}"
        html = browser.get(url)
        candidates = parse.parse_completed_events_list(html, url)
        same_date = [c for c in candidates if c["event_date"] == date]
        if same_date:
            if len(same_date) == 1:
                return same_date[0]
            return max(same_date, key=lambda c: difflib.SequenceMatcher(
                None, c["event_name"].lower(), name.lower()).ratio())
        if candidates and candidates[-1]["event_date"] < date:
            break  # pages go newest -> oldest; past this point everything is older still
    return None


def scrape_event(browser, event_id):
    """One UFCStats event, fully: event meta, every fight (totals +
    round-by-round), every fighter who appeared. Returns (event_meta,
    fight_bundles, fighter_rows) - see merge.build_fight_bundle for the
    bundle shape."""
    event_url = f"http://ufcstats.com/event-details/{event_id}"
    event_page = parse.parse_event_page(browser.get(event_url), event_url)
    event_meta = {
        "event_id": event_id,
        "event_name": event_page["event_name"],
        "event_date": event_page["event_date"],
        "location": event_page["location"],
    }

    bundles, fighter_ids = [], set()
    for event_fight in event_page["fights"]:
        fight_url = f"http://ufcstats.com/fight-details/{event_fight['fight_id']}"
        fight_detail = parse.parse_fight_page(browser.get(fight_url), event_fight["fight_id"], fight_url)
        bundles.append(merge.build_fight_bundle(event_meta, event_fight, fight_detail))
        fighter_ids.update([fight_detail["r_fighter_id"], fight_detail["b_fighter_id"]])

    fighter_rows = []
    for fighter_id in sorted(fighter_ids):
        fighter_url = f"http://ufcstats.com/fighter-details/{fighter_id}"
        fighter_rows.append(parse.parse_fighter_page(browser.get(fighter_url), fighter_id, fighter_url))

    return event_meta, bundles, fighter_rows


def run(data_dir="data", dry_run=False):
    missing = find_missing_events(data_dir)
    if not missing:
        print("[pipeline] no new events on ufc.com/events - nothing to do")
        return {"status": "no_new_events"}

    print(f"[pipeline] {len(missing)} event(s) missing from data/event.csv: "
          + ", ".join(f"{e['name']} ({e['date']})" for e in missing))

    all_event_metas, all_bundles, all_fighter_rows = [], [], []
    with UfcStatsBrowser() as browser:
        for ufc_event in missing:
            match = find_ufcstats_event(browser, ufc_event["name"], ufc_event["date"])
            if match is None:
                print(f"[pipeline] '{ufc_event['name']}' ({ufc_event['date']}) not found yet on "
                      f"UFCStats - skipping this run, will retry next time")
                continue
            print(f"[pipeline] scraping '{match['event_name']}' ({match['event_date']}) "
                  f"from http://ufcstats.com/event-details/{match['event_id']}")
            try:
                event_meta, bundles, fighter_rows = scrape_event(browser, match["event_id"])
            except parse.ParseError as exc:
                raise PipelineError(f"failed scraping '{match['event_name']}': {exc}") from exc
            all_event_metas.append(event_meta)
            all_bundles.extend(bundles)
            all_fighter_rows.extend(fighter_rows)
            print(f"[pipeline]   {len(bundles)} fight(s), "
                  f"{sum(len(b['fight_rounds_rows']) for b in bundles)} round row(s), "
                  f"{len(fighter_rows)} fighter page(s)")

    if not all_bundles:
        print("[pipeline] every missing event was unavailable on UFCStats this run - nothing written")
        return {"status": "no_new_events"}

    existing = merge.load_existing(data_dir)
    updated = dict(existing)

    event_csv_rows = [{"event_id": e["event_id"], "event_name": e["event_name"],
                       "date": e["event_date"], "location": e["location"]} for e in all_event_metas]
    updated["event.csv"], n_events, _ = merge.append_new(updated["event.csv"], event_csv_rows, "event_id")
    updated["fight.csv"], n_fights, _ = merge.append_new(
        updated["fight.csv"], [b["fight_row"] for b in all_bundles], "fight_id")
    updated["fights.csv"], _, _ = merge.append_new(
        updated["fights.csv"], [b["fights_row"] for b in all_bundles], "fight_id")
    round_rows = [row for b in all_bundles for row in b["fight_rounds_rows"]]
    updated["fight_rounds.csv"], n_rounds, _ = merge.append_new(
        updated["fight_rounds.csv"], round_rows, ["fight_id", "round_no"])
    bonus_rows = [row for b in all_bundles for row in b["bonus_rows"]]
    updated["fighter_bonus.csv"], n_bonuses, _ = merge.append_new(
        updated["fighter_bonus.csv"], bonus_rows, ["fight_id", "bonus_type"])
    updated["fighter.csv"], n_new_fighters, n_updated_fighters = merge.upsert_fighters(
        updated["fighter.csv"], all_fighter_rows)

    summary = {
        "status": "dry_run" if dry_run else "written",
        "events": n_events, "fights": n_fights, "rounds": n_rounds,
        "bonuses": n_bonuses, "new_fighters": n_new_fighters, "updated_fighters": n_updated_fighters,
    }
    print(f"[pipeline] {summary}")

    merge.validate_new_data(updated)
    print("[pipeline] validation passed (clean_fights/clean_rounds + integrity checks)")

    if dry_run:
        print("[pipeline] --dry-run: not writing data/*.csv")
        return summary

    merge.write_all(data_dir, updated)
    print(f"[pipeline] wrote data/*.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="scrape and validate, but don't write data/*.csv")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    try:
        result = run(data_dir=args.data_dir, dry_run=args.dry_run)
    except (PipelineError, merge.ValidationError, parse.ParseError, ufc_com.UfcComParseError) as exc:
        print(f"[pipeline] FAILED: {exc}", file=sys.stderr)
        return 1

    return 0 if result["status"] in ("no_new_events", "dry_run", "written") else 1


if __name__ == "__main__":
    sys.exit(main())
