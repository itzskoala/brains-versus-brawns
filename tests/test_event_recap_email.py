import pandas as pd
import pytest

from src.monitoring.event_recap_email import (
    build_event_recap_data,
    build_event_recap_email,
    render_recap_html,
    render_recap_text,
)

EVENT_NAME = "UFC Fight Night: Example vs. Card"
EVENT_LOCATION = "Las Vegas, Nevada, USA"


def make_fight(fight_id, r_id, b_id, winner_id, method="KO/TKO", result_status="win",
               finish_round=2, finish_time=125, rounds_fought=2, details=None,
               title_fight=False, weight_class="Lightweight", event_name=EVENT_NAME):
    return {
        "fight_id": fight_id, "event_name": event_name, "event_date": pd.Timestamp("2026-10-17"),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "result_status": result_status, "method": method, "details": details,
        "finish_round": finish_round, "finish_time": finish_time, "rounds_fought": rounds_fought,
        "weight_class": weight_class, "title_fight": title_fight,
        "r_total_sig_landed": 20, "r_total_sig_atmp": 40,
        "b_total_sig_landed": 15, "b_total_sig_atmp": 30,
        "r_total_td_success": 1, "r_total_td_atmp": 2,
        "b_total_td_success": 0, "b_total_td_atmp": 1,
        "r_total_ctrl_seconds": 60, "b_total_ctrl_seconds": 30,
    }


def make_fighters_df(names):
    # names: {fighter_id: fighter_name}
    return pd.DataFrame({"fighter_id": list(names.keys()), "fighter_name": list(names.values())})


def make_events_df(event_name=EVENT_NAME, location=EVENT_LOCATION):
    return pd.DataFrame([{"event_name": event_name, "location": location}])


NAMES = {
    "main_a": "Joaquin Buckley", "main_b": "Mike Malott",
    "co_a": "Erin Blanchfield", "co_b": "Jasmine Jasudavicius",
    "nc_a": "Daniel Cormier", "nc_b": "Jon Jones",
    "dq_a": "Dean Barry", "dq_b": "Mike Jackson",
    "draw_a": "Damon Jackson", "draw_b": "Levan Makashvili",
    "fotn_a": "Mark Hominick", "fotn_b": "Yves Jabouin",
}
FIGHTERS_DF = make_fighters_df(NAMES)


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

def test_subject_is_exactly_the_event_name_nothing_else():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a")])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    email = build_event_recap_email(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                     fighters_df=FIGHTERS_DF, events=make_events_df())
    assert email["subject"] == EVENT_NAME


# ---------------------------------------------------------------------------
# Complete card, in order
# ---------------------------------------------------------------------------

def test_every_fight_is_listed_main_event_first():
    fights = pd.DataFrame([
        make_fight("f1", "main_a", "main_b", "main_a"),
        make_fight("f2", "co_a", "co_b", "co_b"),
        make_fight("f3", "dq_a", "dq_b", "dq_b", method="DQ", result_status="win"),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    assert [f["fight_id"] for f in data["fights"]] == ["f1", "f2", "f3"]
    assert data["fights"][0]["red_name"] == "Joaquin Buckley"
    assert data["fights"][0]["blue_name"] == "Mike Malott"


def test_every_listed_fight_appears_in_both_text_and_html_output():
    fights = pd.DataFrame([
        make_fight("f1", "main_a", "main_b", "main_a"),
        make_fight("f2", "co_a", "co_b", "co_b"),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    text = render_recap_text(data)
    html = render_recap_html(data)
    for name in ("Joaquin Buckley", "Mike Malott", "Erin Blanchfield", "Jasmine Jasudavicius"):
        assert name in text
        assert name in html


def test_unknown_event_raises_instead_of_returning_an_empty_recap():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a")])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    with pytest.raises(ValueError):
        build_event_recap_data("Not A Real Event", fights=fights, fighter_bonus=bonus,
                                fighters_df=FIGHTERS_DF, events=make_events_df())


# ---------------------------------------------------------------------------
# Header: name, date, location
# ---------------------------------------------------------------------------

def test_header_includes_date_and_location():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a")])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    text = render_recap_text(data)
    assert "October 17, 2026" in text
    assert EVENT_LOCATION in text
    html = render_recap_html(data)
    assert "October 17, 2026" in html
    assert EVENT_LOCATION in html


# ---------------------------------------------------------------------------
# Bonus tags: Fight of the Night shared by both corners; individual bonuses
# attributed only to a confirmed winner, never guessed.
# ---------------------------------------------------------------------------

def test_fight_of_the_night_tags_both_fighters():
    fights = pd.DataFrame([make_fight("f1", "fotn_a", "fotn_b", "fotn_a")])
    bonus = pd.DataFrame([{"fight_id": "f1", "bonus_type": "Fight of the Night"}])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["red_bonus_tags"] == ["[Performance Bonus: Fight of the Night]"]
    assert fight["blue_bonus_tags"] == ["[Performance Bonus: Fight of the Night]"]
    assert data["bonus_summary"][0]["recipients"] == ["Mark Hominick", "Yves Jabouin"]


def test_individual_bonus_tags_only_the_winner():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_b")])  # blue won
    bonus = pd.DataFrame([{"fight_id": "f1", "bonus_type": "Performance of the Night"}])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["red_bonus_tags"] == []
    assert fight["blue_bonus_tags"] == ["[Performance Bonus: Performance of the Night]"]
    assert data["bonus_summary"][0]["recipients"] == ["Mike Malott"]


def test_bonus_tag_appears_directly_beside_the_recipients_name_in_the_card_line():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a")])
    bonus = pd.DataFrame([{"fight_id": "f1", "bonus_type": "Knockout of the Night"}])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    text = render_recap_text(data)
    assert "Joaquin Buckley [Performance Bonus: Knockout of the Night] vs. Mike Malott" in text


def test_individual_bonus_on_a_fight_with_no_decisive_winner_is_never_guessed():
    # Real case: UFC 214's no-contest still had a Performance of the Night
    # bonus on record - there is no confirmed winner to attribute it to, so
    # it must be marked unavailable, not assigned to either fighter.
    fights = pd.DataFrame([
        make_fight("f1", "nc_a", "nc_b", None, method="Overturned", result_status="no_contest",
                   details="Failed Drug Test by Jones"),
    ])
    bonus = pd.DataFrame([{"fight_id": "f1", "bonus_type": "Performance of the Night"}])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["red_bonus_tags"] == []
    assert fight["blue_bonus_tags"] == []
    assert data["bonus_summary"][0]["recipients"] is None

    text = render_recap_text(data)
    assert "recipient not confirmed in data" in text
    assert "Daniel Cormier" not in text.split("Performance Bonuses:")[1].split("Full Card:")[0]
    assert "Jon Jones" not in text.split("Performance Bonuses:")[1].split("Full Card:")[0]


# ---------------------------------------------------------------------------
# Non-standard outcomes: no fabricated winner.
# ---------------------------------------------------------------------------

def test_ufc_214_style_no_contest_shows_no_fabricated_winner():
    fights = pd.DataFrame([
        make_fight("f1", "nc_a", "nc_b", None, method="Overturned", result_status="no_contest",
                   finish_round=3, finish_time=181, details="Failed Drug Test by Jones"),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["outcome_text"] == "No Contest (Overturned) (Failed Drug Test by Jones)"
    assert "Daniel Cormier by" not in fight["outcome_text"]
    assert "Jon Jones by" not in fight["outcome_text"]


def test_draw_shows_no_fabricated_winner_and_states_decision_subtype():
    fights = pd.DataFrame([
        make_fight("f1", "draw_a", "draw_b", None, method="Decision - Majority", result_status="draw",
                   finish_round=3, finish_time=300),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["outcome_text"] == "Draw (Majority Decision)"


def test_dq_shows_the_real_winner_and_the_foul_reason():
    fights = pd.DataFrame([
        make_fight("f1", "dq_a", "dq_b", "dq_b", method="DQ", result_status="win",
                   finish_round=1, finish_time=45, details="Illegal Knee to a Grounded Opponent"),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    fight = data["fights"][0]
    assert fight["outcome_text"] == "Mike Jackson by Disqualification (Illegal Knee to a Grounded Opponent)"


def test_decision_subtypes_are_each_distinguished():
    fights = pd.DataFrame([
        make_fight("f1", "main_a", "main_b", "main_a", method="Decision - Unanimous",
                   finish_round=3, finish_time=300),
        make_fight("f2", "co_a", "co_b", "co_a", method="Decision - Split",
                   finish_round=3, finish_time=300),
    ])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    assert "Unanimous Decision" in data["fights"][0]["outcome_text"]
    assert "Split Decision" in data["fights"][1]["outcome_text"]


# ---------------------------------------------------------------------------
# No model predictions anywhere in the output.
# ---------------------------------------------------------------------------

_PREDICTION_WORDS = ("predict", "probability", "win_probability", "model", "%")


def test_no_prediction_or_probability_language_anywhere_in_the_email():
    fights = pd.DataFrame([
        make_fight("f1", "main_a", "main_b", "main_a"),
        make_fight("f2", "nc_a", "nc_b", None, method="Overturned", result_status="no_contest"),
        make_fight("f3", "draw_a", "draw_b", None, method="Decision - Majority", result_status="draw"),
    ])
    bonus = pd.DataFrame([{"fight_id": "f1", "bonus_type": "Performance of the Night"}])
    email = build_event_recap_email(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                     fighters_df=FIGHTERS_DF, events=make_events_df())
    for body in (email["text_body"], email["html_body"]):
        lowered = body.lower()
        for word in _PREDICTION_WORDS:
            assert word not in lowered, f"found prediction-related text {word!r} in recap output"


def test_build_event_recap_email_result_has_no_prediction_fields():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a")])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    email = build_event_recap_email(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                     fighters_df=FIGHTERS_DF, events=make_events_df())
    assert set(email.keys()) == {"subject", "html_body", "text_body"}


# ---------------------------------------------------------------------------
# Stats and missing-data handling
# ---------------------------------------------------------------------------

def test_stats_line_present_when_round_detail_exists():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a", rounds_fought=2)])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    assert data["fights"][0]["stats_line"] == "Sig. Str: 20/40 vs 15/30 | Takedowns: 1/2 vs 0/1 | Control: 1:00 vs 0:30"


def test_stats_line_omitted_not_fabricated_when_no_round_detail():
    fights = pd.DataFrame([make_fight("f1", "main_a", "main_b", "main_a", rounds_fought=0)])
    bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])
    data = build_event_recap_data(EVENT_NAME, fights=fights, fighter_bonus=bonus,
                                   fighters_df=FIGHTERS_DF, events=make_events_df())
    assert data["fights"][0]["stats_line"] is None
    text = render_recap_text(data)
    assert "Sig. Str" not in text
