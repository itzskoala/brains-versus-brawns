"""Factual, prediction-free results recap for one completed UFC event -
subject line, HTML body, and plain-text fallback. Built entirely from
already-scraped data (data/event.csv, data/fights.csv,
data/fighter_bonus.csv, data/fighter.csv via src.data.cleaning.clean_fights
- the same cleaned shape every other part of this project reads). No model
predictions, win probabilities, or any src.models/src.features.snapshot
import anywhere in this module - this is a results report, not a
prediction report (see api/main.py's past_meetings for that instead).

Usage: build_event_recap_email(event_name) -> {"subject", "html_body",
"text_body"}, ready to hand to an email-sending tool (e.g. the Gmail MCP
connector's send_message: subject=subject, htmlBody=html_body,
body=text_body).
"""
import html as html_lib

import pandas as pd

from src.data.cleaning import clean_fights
from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

# UFCStats' own method text -> a cleaner display phrase. Falls back to the
# raw text unchanged for anything not listed here - never invents a label
# for a method value this map doesn't recognize.
_METHOD_DISPLAY = {
    "KO/TKO": "KO/TKO",
    "TKO - Doctor's Stoppage": "TKO (Doctor's Stoppage)",
    "Submission": "Submission",
    "Decision - Unanimous": "Unanimous Decision",
    "Decision - Split": "Split Decision",
    "Decision - Majority": "Majority Decision",
    "Decision": "Decision",
    "DQ": "Disqualification",
    "Overturned": "Overturned",
    "Could Not Continue": "Could Not Continue",
    "Other": "Other",
}

# Bonus types UFCStats records per fight (src/data/scraping/ufcstats_parse.py
# BONUS_ORDER) - Fight of the Night is awarded to the fight itself (both
# corners); every other type is awarded to exactly one fighter.
_SHARED_BONUS_TYPES = {"Fight of the Night"}


def _method_display(method):
    if pd.isna(method):
        return None
    return _METHOD_DISPLAY.get(method, method)


def _format_seconds(seconds):
    """123 -> "2:03". None/NaN -> None (never a fabricated "0:00")."""
    if seconds is None or pd.isna(seconds):
        return None
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def _load_cleaned_fights():
    master = build_master_equivalent(load_fights(), load_fighters())
    return clean_fights(master)


def _bonus_recipients_by_fight(bonus_rows_for_event, fights_by_id):
    """{fight_id: [{"bonus_type", "fighter_ids": [...]}]}.

    Fight of the Night: both corners (fighter_ids has both r_fighter_id and
    b_fighter_id) - structurally the only possible reading, since the award
    is given to the fight, not an individual.

    Every other bonus type (Performance/Knockout/Submission of the Night):
    UFCStats' own event page records these bonus icons at the FIGHT level,
    not per-fighter (confirmed in src/data/scraping/ufcstats_parse.py -
    "NOT corner info" - and src/data/scraping/merge.py), so there is no
    scraped field naming which specific fighter earned it. This is not
    guessed: UFC's own award rules mean a fight can only ever produce a
    legitimate individual-bonus winner, never the loser of that same fight
    - so the fight's own CONFIRMED winner_id is the recipient whenever one
    exists (result_status == "win"). If a fight has no decisive winner (a
    draw/no-contest carrying one of these bonus types - not observed in
    this dataset, but handled defensively rather than guessed), fighter_ids
    is left empty and the caller must mark it unavailable rather than
    attribute it to anyone."""
    by_fight = {}
    for _, bonus_row in bonus_rows_for_event.iterrows():
        fight_id = bonus_row["fight_id"]
        bonus_type = bonus_row["bonus_type"]
        fight = fights_by_id.get(fight_id)
        if fight is None:
            continue  # bonus recorded for a fight outside this event - skip

        if bonus_type in _SHARED_BONUS_TYPES:
            fighter_ids = [fight["r_fighter_id"], fight["b_fighter_id"]]
        elif fight["result_status"] == "win" and pd.notna(fight["winner_id"]):
            fighter_ids = [fight["winner_id"]]
        else:
            fighter_ids = []  # no confirmed recipient - never guessed

        by_fight.setdefault(fight_id, []).append({"bonus_type": bonus_type, "fighter_ids": fighter_ids})
    return by_fight


def _fight_stats_line(fight):
    """Sig. str / takedowns / control time for both corners, or None if
    this fight has no round-by-round detail at all (rounds_fought == 0 -
    a missingness marker, see src/features/engineering.py's
    _prior_career_totals docstring - never shown as a fabricated 0."""
    if not fight["rounds_fought"] > 0:
        return None

    def corner(prefix):
        sig = f"{int(fight[f'{prefix}_total_sig_landed'])}/{int(fight[f'{prefix}_total_sig_atmp'])}"
        td = f"{int(fight[f'{prefix}_total_td_success'])}/{int(fight[f'{prefix}_total_td_atmp'])}"
        ctrl = _format_seconds(fight[f"{prefix}_total_ctrl_seconds"]) or "0:00"
        return sig, td, ctrl

    r_sig, r_td, r_ctrl = corner("r")
    b_sig, b_td, b_ctrl = corner("b")
    return (f"Sig. Str: {r_sig} vs {b_sig} | Takedowns: {r_td} vs {b_td} | "
            f"Control: {r_ctrl} vs {b_ctrl}")


def _outcome_text(fight, r_name, b_name):
    """"<Winner> by <Method>" / "Draw (<Method>)" / "No Contest (<Method>)"
    - never a fabricated winner for a draw/no-contest. Appends the
    recorded details (foul/overturn reason) in parentheses when present."""
    method = _method_display(fight["method"])
    # Collapse embedded newlines/extra whitespace from the raw scraped text
    # (e.g. a multi-line judges'-scorecard string) into one line - a
    # presentation cleanup, not a change to the content itself.
    details = " ".join((fight["details"] or "").split())
    suffix = f" ({details})" if details else ""

    if fight["result_status"] == "draw":
        return f"Draw ({method}){suffix}" if method else f"Draw{suffix}"
    if fight["result_status"] == "no_contest":
        return f"No Contest ({method}){suffix}" if method else f"No Contest{suffix}"

    winner_name = r_name if fight["winner_id"] == fight["r_fighter_id"] else b_name
    return f"{winner_name} by {method}{suffix}" if method else f"{winner_name}{suffix}"


def _bonus_tag(bonus_type):
    return f"[Performance Bonus: {bonus_type}]"


def build_event_recap_data(event_name, fights=None, fighter_bonus=None, fighters_df=None, events=None):
    """Assemble every fact this recap needs for one event, as plain data
    (no HTML/text formatting yet - see render_recap_html/render_recap_text).
    fights/fighter_bonus/fighters_df are injectable for tests; default to
    the real data/*.csv files via the same loaders/cleaning every other
    part of this project uses.

    Raises ValueError if event_name has no fights on record - never
    returns a recap for an event that doesn't exist.

    Card order (main event first, through the preliminaries) is the order
    fights already have in data/fights.csv for that event: UFCStats' own
    event page lists fights in that order, and the scraper
    (src/data/scraping/ufcstats_parse.py parse_event_page) preserves it -
    confirmed against real data (Buckley vs. Malott's actual main event is
    the first row for that event_name)."""
    fighters_df = load_fighters() if fighters_df is None else fighters_df
    fights = _load_cleaned_fights() if fights is None else fights
    if fighter_bonus is None:
        fighter_bonus = pd.read_csv("data/fighter_bonus.csv")
    if events is None:
        events = pd.read_csv("data/event.csv")

    event_fights = fights[fights["event_name"] == event_name]
    if event_fights.empty:
        raise ValueError(f"no fights on record for event {event_name!r}")

    # event_location (unlike event_date) isn't on the cleaned fights frame -
    # clean_fights drops it (src/data/cleaning.py), so it's read from
    # data/event.csv directly instead, the canonical source for it anyway.
    event_row = events[events["event_name"] == event_name]
    event_date = event_fights["event_date"].iloc[0]
    event_location = event_row["location"].iloc[0] if not event_row.empty else None

    names = fighters_df.set_index("fighter_id")["fighter_name"]
    fights_by_id = {row["fight_id"]: row for _, row in event_fights.iterrows()}
    bonus_rows_for_event = fighter_bonus[fighter_bonus["fight_id"].isin(fights_by_id.keys())]
    bonuses_by_fight = _bonus_recipients_by_fight(bonus_rows_for_event, fights_by_id)

    card = []
    bonus_summary = []
    for _, fight in event_fights.iterrows():
        r_name = names.get(fight["r_fighter_id"], fight["r_fighter_id"])
        b_name = names.get(fight["b_fighter_id"], fight["b_fighter_id"])

        r_tags, b_tags = [], []
        for bonus in bonuses_by_fight.get(fight["fight_id"], []):
            tag = _bonus_tag(bonus["bonus_type"])
            recipient_names = []
            if fight["r_fighter_id"] in bonus["fighter_ids"]:
                r_tags.append(tag)
                recipient_names.append(r_name)
            if fight["b_fighter_id"] in bonus["fighter_ids"]:
                b_tags.append(tag)
                recipient_names.append(b_name)
            bonus_summary.append({
                "bonus_type": bonus["bonus_type"],
                "recipients": recipient_names or None,  # None -> unconfirmed, never guessed
            })

        card.append({
            "fight_id": fight["fight_id"],
            "red_name": r_name, "blue_name": b_name,
            "red_bonus_tags": r_tags, "blue_bonus_tags": b_tags,
            "weight_class": fight["weight_class"],
            "title_fight": bool(fight["title_fight"]),
            "outcome_text": _outcome_text(fight, r_name, b_name),
            "finish_round": int(fight["finish_round"]) if pd.notna(fight["finish_round"]) else None,
            "finish_time": _format_seconds(fight["finish_time"]),
            "stats_line": _fight_stats_line(fight),
        })

    return {
        "event_name": event_name,
        "event_date": event_date,
        "event_location": event_location,
        "fights": card,
        "bonus_summary": bonus_summary,
    }


def _fight_line_text(fight):
    red = fight["red_name"] + ("".join(f" {t}" for t in fight["red_bonus_tags"]))
    blue = fight["blue_name"] + ("".join(f" {t}" for t in fight["blue_bonus_tags"]))
    when = f"Round {fight['finish_round']}, {fight['finish_time']}" if fight["finish_time"] else \
        (f"Round {fight['finish_round']}" if fight["finish_round"] else "")
    title = " (Title Fight)" if fight["title_fight"] else ""
    line = f"{red} vs. {blue}{title} — {fight['outcome_text']}"
    if when:
        line += f" — {when}"
    return line


def render_recap_text(data):
    lines = [data["event_name"]]
    header = data["event_date"].strftime("%B %d, %Y") if pd.notna(data["event_date"]) else "Date unavailable"
    if data["event_location"]:
        header += f" — {data['event_location']}"
    lines.append(header)
    lines.append("")

    if data["bonus_summary"]:
        lines.append("Performance Bonuses:")
        for bonus in data["bonus_summary"]:
            recipients = ", ".join(bonus["recipients"]) if bonus["recipients"] else "recipient not confirmed in data"
            lines.append(f"  - {bonus['bonus_type']}: {recipients}")
        lines.append("")

    lines.append("Full Card:")
    for fight in data["fights"]:
        lines.append(_fight_line_text(fight))
        if fight["stats_line"]:
            lines.append(f"    {fight['stats_line']}")
    return "\n".join(lines)


def render_recap_html(data):
    event_date = data["event_date"].strftime("%B %d, %Y") if pd.notna(data["event_date"]) else "Date unavailable"
    header = html_lib.escape(event_date)
    if data["event_location"]:
        header += " &mdash; " + html_lib.escape(data["event_location"])

    parts = [
        f"<h1>{html_lib.escape(data['event_name'])}</h1>",
        f"<p>{header}</p>",
    ]

    if data["bonus_summary"]:
        parts.append("<h2>Performance Bonuses</h2><ul>")
        for bonus in data["bonus_summary"]:
            recipients = ", ".join(bonus["recipients"]) if bonus["recipients"] else "recipient not confirmed in data"
            parts.append(f"<li><strong>{html_lib.escape(bonus['bonus_type'])}:</strong> "
                         f"{html_lib.escape(recipients)}</li>")
        parts.append("</ul>")

    parts.append("<h2>Full Card</h2><ol>")
    for fight in data["fights"]:
        red = html_lib.escape(fight["red_name"])
        for tag in fight["red_bonus_tags"]:
            red += f" <em>{html_lib.escape(tag)}</em>"
        blue = html_lib.escape(fight["blue_name"])
        for tag in fight["blue_bonus_tags"]:
            blue += f" <em>{html_lib.escape(tag)}</em>"
        title = " <strong>(Title Fight)</strong>" if fight["title_fight"] else ""
        when = f"Round {fight['finish_round']}, {fight['finish_time']}" if fight["finish_time"] else \
            (f"Round {fight['finish_round']}" if fight["finish_round"] else "")

        parts.append("<li>")
        parts.append(f"{red} vs. {blue}{title}<br>")
        parts.append(f"{html_lib.escape(fight['outcome_text'])}" + (f" &mdash; {when}" if when else ""))
        if fight["stats_line"]:
            parts.append(f"<br><small>{html_lib.escape(fight['stats_line'])}</small>")
        parts.append("</li>")
    parts.append("</ol>")

    return "\n".join(parts)


def build_event_recap_email(event_name, fights=None, fighter_bonus=None, fighters_df=None, events=None):
    """{"subject", "html_body", "text_body"} for one event - subject is
    exactly event_name, nothing else. No model predictions/probabilities
    anywhere in the output."""
    data = build_event_recap_data(event_name, fights=fights, fighter_bonus=fighter_bonus,
                                   fighters_df=fighters_df, events=events)
    return {
        "subject": event_name,
        "html_body": render_recap_html(data),
        "text_body": render_recap_text(data),
    }
