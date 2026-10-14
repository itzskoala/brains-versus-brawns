"""Pure HTML -> data parsing for UFCStats.com pages. Every function here
takes an already-fetched HTML string (see ufcstats_browser.py for how that
HTML gets rendered) and returns plain dicts/lists - no network, no
Playwright - so these are fully unit-testable against saved fixtures,
including deliberately broken ones (the challenge-page shell, truncated
markup, a fight with no round tables).

Output field names match data/fight.csv, data/fights.csv,
data/fight_rounds.csv, data/fighter.csv, and data/fighter_bonus.csv exactly
(src.data.scraping.merge does the CSV-shaping); verified against real
UFCStats pages fetched 2026-10-14 (UFC 330: Makhachev vs. Machado Garry and
its fight/fighter pages).
"""
import re
from datetime import datetime

from bs4 import BeautifulSoup

# event-page weight-class-cell icon filename -> what it means. belt.png is
# handled separately as the title_fight flag; the rest are fighter_bonus.csv
# bonus_type values, in the same order UFCStats' own "bonuses" text joins
# them with (verified against data/fights.csv's "bonuses" column, e.g.
# "Fight of the Night, Performance of the Night").
BONUS_ICON_MAP = {
    "fight.png": "Fight of the Night",
    "ko.png": "Knockout of the Night",
    "perf.png": "Performance of the Night",
    "sub.png": "Submission of the Night",
}
BONUS_ORDER = ["Fight of the Night", "Knockout of the Night",
               "Performance of the Night", "Submission of the Night"]


class ParseError(Exception):
    """A UFCStats page didn't have the structure this parser expects."""


class ChallengePageError(ParseError):
    """The page is still the "Checking your browser..." bot-check shell,
    not real content - the browser layer should have resolved this; seeing
    it here means something upstream broke."""


def _require(soup, selector, url, what):
    el = soup.select_one(selector)
    if el is None:
        text = soup.get_text(" ", strip=True)
        if "checking your browser" in text.lower():
            raise ChallengePageError(f"{url}: still the bot-check challenge page, not {what}")
        raise ParseError(f"{url}: expected {selector!r} ({what}) not found - markup may have changed")
    return el


def _id_from_url(url):
    return url.rstrip("/").rsplit("/", 1)[-1]


def _landed_of_attempted(text):
    """'13 of 56' -> (13, 56). '---' / '' -> (None, None) - some columns
    (TD, Rev, Sub, early fights) use that placeholder for "not applicable"."""
    text = text.strip()
    m = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _int_or_none(text):
    text = text.strip()
    return int(text) if text.isdigit() else None


def _ctrl_text_or_none(text):
    """UFCStats' own 'mm:ss' control-time text, or None for '--' (not
    tracked) - matches fight_rounds.csv's r_ctrl/b_ctrl, which stores this
    same raw text (src.data.cleaning.clean_rounds converts it to seconds)."""
    text = text.strip()
    return text if re.match(r"\d+:\d{2}$", text) else None


def _ctrl_seconds_or_none(text):
    """Same field, converted to seconds - matches fights.csv's
    r_total_ctrl_seconds/b_total_ctrl_seconds, which (unlike
    fight_rounds.csv) stores the total as an int."""
    text = _ctrl_text_or_none(text)
    if text is None:
        return None
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + int(seconds)


def parse_completed_events_list(html, url=""):
    """UFCStats' own http://ufcstats.com/statistics/events/completed (and
    its ?page=N continuations). Returns events newest-first:
    [{"event_id", "event_name", "event_date" (date), "location"}, ...]."""
    soup = BeautifulSoup(html, "lxml")
    table = _require(soup, "table.b-statistics__table-events", url, "the completed-events table")

    events = []
    for row in table.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date_span = row.select_one("span.b-statistics__date")
        if link is None or date_span is None:
            continue  # header row / spacer row
        cols = row.select("td.b-statistics__table-col")
        location = " ".join(cols[1].get_text(" ", strip=True).split()) if len(cols) > 1 else None
        events.append({
            "event_id": _id_from_url(link["href"]),
            "event_name": " ".join(link.get_text(" ", strip=True).split()),
            "event_date": datetime.strptime(date_span.get_text(strip=True), "%B %d, %Y").date(),
            "location": location,
        })

    if not events:
        raise ParseError(f"{url}: events table found but zero rows parsed - markup may have changed")
    return events


def parse_event_page(html, url=""):
    """One UFCStats event-details page. Returns the event's own
    name/date/location plus one entry per fight with everything only this
    page has (weight class, title-fight flag, bonuses):
    {"event_name", "event_date", "location",
     "fights": [{"fight_id", "weight_class", "title_fight" (bool),
                 "bonus_types" (list[str])}, ...]}

    Deliberately does NOT return which fighter is which corner here, even
    though the fight-list table lists two fighter links per row: checked
    against two already-known historical fights, this table's fighter
    order does not reliably match data/fight.csv's r_id/b_id (one of the
    two agreed, one didn't), while parse_fight_page's persons-block order
    matched both. parse_fight_page is the only authoritative source for
    r_fighter_id/b_fighter_id/winner_id - callers must get corner
    assignment from there, and only use this page for the fight-level
    facts (weight class, title fight, bonuses) that aren't corner-specific
    anyway."""
    soup = BeautifulSoup(html, "lxml")
    title = _require(soup, "span.b-content__title-highlight", url, "the event title")
    event_name = " ".join(title.get_text(" ", strip=True).split())

    event_date, location = None, None
    for item in soup.select("li.b-list__box-list-item"):
        label_el = item.select_one("i.b-list__box-item-title")
        if label_el is None:
            continue
        label = label_el.get_text(strip=True)
        full = item.get_text(" ", strip=True)
        value = full[len(label):].strip() if full.startswith(label) else full
        if label == "Date:":
            event_date = datetime.strptime(value, "%B %d, %Y").date()
        elif label == "Location:":
            location = value

    fight_rows = soup.select("tr[data-link]")
    if not fight_rows:
        raise ParseError(f"{url}: event title/date found but zero fight rows - markup may have changed")

    fights = []
    for row in fight_rows:
        tds = row.find_all("td")
        if len(tds) < 7:
            raise ParseError(f"{url}: fight row has {len(tds)} columns, expected >= 7")

        weight_cell = tds[6]
        icons = {img.get("src", "").rsplit("/", 1)[-1] for img in weight_cell.find_all("img")}
        bonus_types = [label for icon, label in BONUS_ICON_MAP.items() if icon in icons]
        bonus_types = [b for b in BONUS_ORDER if b in bonus_types]
        weight_class = weight_cell.find("p").get_text(" ", strip=True).split("\n")[0].strip()
        # strip trailing residue from the belt/bonus <img> tags rendering inline
        weight_class = re.sub(r"\s+", " ", weight_class).strip()

        fights.append({
            "fight_id": _id_from_url(row["data-link"]),
            "weight_class": weight_class,
            "title_fight": "belt.png" in icons,
            "bonus_types": bonus_types,
        })

    return {"event_name": event_name, "event_date": event_date, "location": location, "fights": fights}


def _parse_persons(soup, url):
    persons = soup.select("div.b-fight-details__person")
    if len(persons) != 2:
        raise ParseError(f"{url}: found {len(persons)} fighters in persons block, expected 2")

    out = []
    for person in persons:
        status = person.select_one(".b-fight-details__person-status")
        link = person.select_one("a.b-fight-details__person-link")
        if status is None or link is None:
            raise ParseError(f"{url}: a fighter block is missing its status or link")
        out.append({
            "fighter_id": _id_from_url(link["href"]),
            "status": status.get_text(strip=True).upper(),
        })
    return out


def _parse_fight_head(soup, url):
    fields = {}
    for item in soup.select("i.b-fight-details__text-item, i.b-fight-details__text-item_first"):
        label_el = item.select_one("i.b-fight-details__label")
        if label_el is None:
            continue
        label = label_el.get_text(strip=True)
        full = item.get_text(" ", strip=True)
        value = full[len(label):].strip() if full.startswith(label) else full
        fields[label] = value

    details_label = soup.find("i", class_="b-fight-details__label", string=lambda s: s and "Details" in s)
    details = ""
    if details_label is not None:
        details_p = details_label.find_parent("p")
        if details_p is not None:
            full = details_p.get_text(" ", strip=True)
            details = full[len("Details:"):].strip() if full.startswith("Details:") else full

    required = ["Method:", "Round:", "Time:", "Time format:"]
    missing = [r for r in required if r not in fields]
    if missing:
        raise ParseError(f"{url}: fight-head missing field(s) {missing}")

    return {
        "method": fields["Method:"],
        "round": int(fields["Round:"]),
        "time": fields["Time:"],
        "time_format": fields["Time format:"],
        "referee": fields.get("Referee:") or None,
        "details": details,
    }


# (fights.csv total column, fight_rounds.csv column) pairs, in the exact
# left-to-right order UFCStats renders the Totals/per-round-totals table
# columns (after the Fighter column): KD, Sig.str, Sig.str% (skipped),
# Total str, TD, TD% (skipped), Sub.att, Rev, Ctrl.
_TOTALS_COLUMNS = ["kd", "sig", None, "total_str", "td", None, "sub_att", "rev", "ctrl"]
# Sig-strikes-by-target table columns, after Fighter: Sig.str (skipped,
# duplicate of the totals table), Sig.str% (skipped), Head, Body, Leg,
# Distance, Clinch, Ground.
_TARGET_COLUMNS = [None, None, "head", "body", "leg", "distance", "clinch", "ground"]


def _read_stat_row(tds, column_names):
    """One <tr>'s <td>s (after the Fighter column) -> {col_name: (r_value, b_value)}.
    Ctrl columns come back as raw text (mm:ss or None); everything else as
    (landed, attempted) int pairs, except KD/Sub.att/Rev which are plain ints."""
    out = {}
    for name, td in zip(column_names, tds):
        if name is None:
            continue
        ps = td.find_all("p")
        if len(ps) != 2:
            raise ParseError(f"stat column {name!r} has {len(ps)} value(s), expected 2 (one per corner)")
        r_text, b_text = (p.get_text(" ", strip=True) for p in ps)
        if name == "ctrl":
            out[name] = (_ctrl_text_or_none(r_text), _ctrl_text_or_none(b_text))
        elif name in ("kd", "sub_att", "rev"):
            out[name] = (_int_or_none(r_text) or 0, _int_or_none(b_text) or 0)
        else:
            out[name] = (_landed_of_attempted(r_text), _landed_of_attempted(b_text))
    return out


def _totals_row_to_fight_totals(stats, target_stats):
    """One Totals-table row + one sig-strikes-by-target row -> the
    r_total_*/b_total_* fields fights.csv expects."""
    kd_r, kd_b = stats["kd"]
    sig_r, sig_b = stats["sig"]
    total_r, total_b = stats["total_str"]
    td_r, td_b = stats["td"]
    sub_r, sub_b = stats["sub_att"]
    rev_r, rev_b = stats["rev"]
    ctrl_r_text, ctrl_b_text = stats["ctrl"]

    fields = {
        "r_total_kd": kd_r, "b_total_kd": kd_b,
        "r_total_sig_landed": sig_r[0], "r_total_sig_atmp": sig_r[1],
        "b_total_sig_landed": sig_b[0], "b_total_sig_atmp": sig_b[1],
        "r_total_total_str_landed": total_r[0], "r_total_total_str_atmp": total_r[1],
        "b_total_total_str_landed": total_b[0], "b_total_total_str_atmp": total_b[1],
        "r_total_td_success": td_r[0], "r_total_td_atmp": td_r[1],
        "b_total_td_success": td_b[0], "b_total_td_atmp": td_b[1],
        "r_total_sub_att": sub_r, "b_total_sub_att": sub_b,
        "r_total_rev": rev_r, "b_total_rev": rev_b,
        "r_total_ctrl_seconds": _ctrl_seconds_or_none(ctrl_r_text) or 0,
        "b_total_ctrl_seconds": _ctrl_seconds_or_none(ctrl_b_text) or 0,
    }
    for target in ("head", "body", "leg", "distance", "clinch", "ground"):
        r_pair, b_pair = target_stats[target]
        fields[f"r_total_sig_str_landed_{target}"] = r_pair[0]
        fields[f"r_total_sig_str_atmp_{target}"] = r_pair[1]
        fields[f"b_total_sig_str_landed_{target}"] = b_pair[0]
        fields[f"b_total_sig_str_atmp_{target}"] = b_pair[1]
    return fields


def _round_row_to_fight_rounds(round_no, stats, target_stats):
    kd_r, kd_b = stats["kd"]
    sig_r, sig_b = stats["sig"]
    total_r, total_b = stats["total_str"]
    td_r, td_b = stats["td"]
    sub_r, sub_b = stats["sub_att"]
    rev_r, rev_b = stats["rev"]
    ctrl_r, ctrl_b = stats["ctrl"]

    fields = {
        "round_no": round_no,
        "r_kd": kd_r, "b_kd": kd_b,
        "r_sig_landed": sig_r[0], "r_sig_atmp": sig_r[1],
        "b_sig_landed": sig_b[0], "b_sig_atmp": sig_b[1],
        "r_total_str_landed": total_r[0], "r_total_str_atmp": total_r[1],
        "b_total_str_landed": total_b[0], "b_total_str_atmp": total_b[1],
        "r_td_success": td_r[0], "r_td_atmp": td_r[1],
        "b_td_success": td_b[0], "b_td_atmp": td_b[1],
        "r_sub_att": sub_r, "b_sub_att": sub_b,
        "r_rev": rev_r, "b_rev": rev_b,
        "r_ctrl": ctrl_r, "b_ctrl": ctrl_b,
    }
    for target in ("head", "body", "leg", "distance", "clinch", "ground"):
        r_pair, b_pair = target_stats[target]
        fields[f"r_sig_str_landed_{target}"] = r_pair[0]
        fields[f"r_sig_str_atmp_{target}"] = r_pair[1]
        fields[f"b_sig_str_landed_{target}"] = b_pair[0]
        fields[f"b_sig_str_atmp_{target}"] = b_pair[1]
    return fields


def _per_round_rows(table, column_names):
    """A per-round table's body -> {round_no: stats_dict}, reading the
    "Round N" separator rows UFCStats interleaves with each data row. Both
    the table's own column-header row and each "Round N" separator are
    single <th>-cell rows (no <td>s) - a lone <th> with "Round" in its text
    is the separator; anything else that shape is the header, skipped."""
    rows = table.find_all("tr")
    out = {}
    current_round = None
    for row in rows:
        tds = row.find_all("td")
        if not tds:
            ths = row.find_all("th")
            if len(ths) == 1:
                m = re.search(r"Round\s+(\d+)", ths[0].get_text(strip=True))
                if m:
                    current_round = int(m.group(1))
            continue
        if current_round is None:
            raise ParseError("per-round table has a data row before any 'Round N' marker")
        out[current_round] = _read_stat_row(tds[1:], column_names)
    return out


def parse_fight_page(html, fight_id, url=""):
    """One UFCStats fight-details page. Returns everything parse_event_page
    doesn't carry: {"fight_id", "r_fighter_id", "b_fighter_id", "winner_id"
    (or None for a draw/no-contest), "result_status", "method", "round",
    "time", "time_format", "referee", "details", "totals" (r_total_*/
    b_total_* fields for fights.csv), "round_rows" (list of fight_rounds.csv
    rows, oldest round first; empty if this fight has no round-by-round
    tables - some fights simply don't, which is a known, historically
    tolerated gap (see data/unused/scrape_error.csv), not a parse failure)}.
    """
    soup = BeautifulSoup(html, "lxml")
    _require(soup, "div.b-fight-details__persons", url, "the fighters block")

    persons = _parse_persons(soup, url)
    head = _parse_fight_head(soup, url)

    statuses = {p["status"] for p in persons}
    if "W" in statuses:
        winner_id = next(p["fighter_id"] for p in persons if p["status"] == "W")
        result_status = "win"
    elif statuses == {"D"}:
        winner_id, result_status = None, "draw"
    else:
        winner_id, result_status = None, "no_contest"

    # The fight-level "Totals" and "Significant Strikes" tables carry no
    # distinguishing class (only their per-round counterparts do, via
    # "js-fight-table") - so table order is the only reliable signal.
    # UFCStats renders either [totals, totals-per-round, sig-by-target,
    # sig-by-target-per-round] (fight has round data) or just
    # [totals, sig-by-target] (it doesn't - a known, historically tolerated
    # gap, see data/unused/scrape_error.csv's "found 0" round tables).
    tables = soup.find_all("table")
    if len(tables) == 4:
        totals_table, per_round_totals_table, target_table, per_round_target_table = tables
    elif len(tables) == 2:
        totals_table, target_table = tables
        per_round_totals_table = per_round_target_table = None
    else:
        raise ParseError(f"{url}: expected 2 or 4 stat tables, found {len(tables)}")

    def _single_data_row(table, label):
        # UFCStats' <tbody> for these tables is sometimes empty in the
        # parsed DOM with its <tr>s as siblings instead (upstream markup
        # quirk) - a flat find_all("tr") is the only reliable way to reach
        # them; non-data rows (the <th> header) have zero <td>s and are
        # filtered out below.
        rows = [r for r in table.find_all("tr") if len(r.find_all("td")) > 1]
        if len(rows) != 1:
            raise ParseError(f"{url}: {label} table has {len(rows)} data row(s), expected 1")
        return rows[0].find_all("td")[1:]

    totals_stats = _read_stat_row(_single_data_row(totals_table, "totals"), _TOTALS_COLUMNS)
    target_totals_stats = _read_stat_row(_single_data_row(target_table, "sig-strikes-by-target"), _TARGET_COLUMNS)
    totals = _totals_row_to_fight_totals(totals_stats, target_totals_stats)

    round_rows = []
    if per_round_totals_table is not None:
        per_round_totals = _per_round_rows(per_round_totals_table, _TOTALS_COLUMNS)
        per_round_targets = _per_round_rows(per_round_target_table, _TARGET_COLUMNS)
        for round_no in sorted(per_round_totals):
            if round_no not in per_round_targets:
                raise ParseError(f"{url}: round {round_no} has totals but no sig-strikes-by-target row")
            round_rows.append(_round_row_to_fight_rounds(
                round_no, per_round_totals[round_no], per_round_targets[round_no]))

    return {
        "fight_id": fight_id,
        "r_fighter_id": persons[0]["fighter_id"],
        "b_fighter_id": persons[1]["fighter_id"],
        "winner_id": winner_id,
        "result_status": result_status,
        "method": head["method"],
        "round": head["round"],
        "time": head["time"],
        "time_format": head["time_format"],
        "referee": head["referee"],
        "details": head["details"],
        "totals": totals,
        "round_rows": round_rows,
    }


_FIGHTER_LABEL_MAP = {
    "Height:": "height", "Weight:": "weight_lbs", "Reach:": "reach_inches",
    "STANCE:": "stance", "DOB:": "dob", "SLpM:": "slpm", "Str. Acc.:": "str_acc",
    "SApM:": "sapm", "Str. Def:": "str_def", "TD Avg.:": "td_avg",
    "TD Acc.:": "td_acc", "TD Def.:": "td_def", "Sub. Avg.:": "sub_avg",
}


def parse_fighter_page(html, fighter_id, url=""):
    """One UFCStats fighter-details page -> a data/fighter.csv row (as a
    dict). Bio fields and the eight career-rate stats are both read here -
    a returning fighter's row gets fully overwritten with these, matching
    "as of latest scrape" semantics (src/data/ingestion.py's own docstring);
    a debuting fighter gets a brand-new row."""
    soup = BeautifulSoup(html, "lxml")
    name_el = _require(soup, "span.b-content__title-highlight", url, "the fighter name")
    nick_el = soup.select_one("p.b-content__Nickname")

    raw = {}
    for item in soup.select("li.b-list__box-list-item"):
        label_el = item.select_one("i.b-list__box-item-title")
        if label_el is None:
            continue
        label = label_el.get_text(strip=True)
        key = _FIGHTER_LABEL_MAP.get(label)
        if key is None:
            continue
        full = item.get_text(" ", strip=True)
        value = full[len(label):].strip() if full.startswith(label) else full
        raw[key] = value if value and value != "--" else None

    def pct(key):
        v = raw.get(key)
        return int(v.rstrip("%")) if v and v.rstrip("%").isdigit() else None

    def num(key):
        v = raw.get(key)
        return float(v) if v else None

    dob = None
    if raw.get("dob"):
        dob = datetime.strptime(raw["dob"], "%b %d, %Y").date().isoformat()

    weight_lbs = None
    if raw.get("weight_lbs"):
        m = re.match(r"(\d+)", raw["weight_lbs"])
        weight_lbs = int(m.group(1)) if m else None

    reach_inches = None
    if raw.get("reach_inches"):
        m = re.match(r"(\d+)", raw["reach_inches"])
        reach_inches = int(m.group(1)) if m else None

    return {
        "fighter_id": fighter_id,
        "fighter_name": " ".join(name_el.get_text(" ", strip=True).split()),
        "fighter_nick_name": nick_el.get_text(strip=True) if nick_el and nick_el.get_text(strip=True) else None,
        "height": raw.get("height"),
        "weight_lbs": weight_lbs,
        "reach_inches": reach_inches,
        "stance": raw.get("stance"),
        "dob": dob,
        "slpm": num("slpm") or 0.0,
        "str_acc": pct("str_acc") or 0,
        "sapm": num("sapm") or 0.0,
        "str_def": pct("str_def") or 0,
        "td_avg": num("td_avg") or 0.0,
        "td_acc": pct("td_acc") or 0,
        "td_def": pct("td_def") or 0,
        "sub_avg": num("sub_avg") or 0.0,
    }
