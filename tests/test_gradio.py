import pytest
import requests

from frontend import gradio as fg


def test_division_label_lbs_default():
    assert fg._division_label("Welterweight", "lbs") == "Welterweight (170 lbs)"


def test_division_label_kg_uses_published_limit_not_a_conversion():
    # 170 lbs * 0.453592 = 77.11 -> would round to "77 kg" if computed; the
    # UFC's own published limit is 77.1kg - these must match the lookup
    # table (src of truth), not a formula.
    assert fg._division_label("Welterweight", "kg") == "Welterweight (77.1 kg)"
    assert fg._division_label("Heavyweight", "kg") == "Heavyweight (120.2 kg)"


def test_division_label_unknown_division_falls_back_to_bare_name():
    assert fg._division_label("Catchweight", "lbs") == "Catchweight"
    assert fg._division_label("Catchweight", "kg") == "Catchweight"


def test_division_choices_relabels_all_divisions_on_unit_change():
    divisions = ["Flyweight", "Lightweight", "Heavyweight"]
    lbs_choices = fg._division_choices(divisions, "lbs")
    kg_choices = fg._division_choices(divisions, "kg")
    assert [v for _, v in lbs_choices] == divisions  # values (sent to the API) never change
    assert [v for _, v in kg_choices] == divisions
    assert lbs_choices[0][0] == "Flyweight (125 lbs)"
    assert kg_choices[0][0] == "Flyweight (56.7 kg)"


def test_format_height_lbs_is_feet_and_inches_not_bare_inches():
    assert fg._format_height(70, "lbs") == "5'10\""
    assert fg._format_height(72, "lbs") == "6'0\""


def test_format_height_kg_is_centimeters():
    assert fg._format_height(70, "kg") == "178 cm"


def test_format_reach_lbs_is_inches():
    assert fg._format_reach(74, "lbs") == '74"'


def test_format_reach_kg_is_centimeters():
    assert fg._format_reach(74, "kg") == "188 cm"


def test_format_height_and_reach_missing_value_is_dash_not_invented():
    assert fg._format_height(None, "lbs") == "-"
    assert fg._format_height(None, "kg") == "-"
    assert fg._format_reach(None, "lbs") == "-"


def test_split_name_two_word_name():
    assert fg._split_name("Sean O'Malley") == ("Sean", "O'Malley")


def test_split_name_three_word_name_keeps_remainder_as_last():
    assert fg._split_name("Ian Machado Garry") == ("Ian", "Machado Garry")


def test_split_name_single_word_name():
    assert fg._split_name("Shogun") == ("Shogun", "")


def test_format_fighter_name_with_nickname():
    assert fg._format_fighter_name("Sean O'Malley", "Suga") == 'Sean "Suga" O\'Malley'


def test_format_fighter_name_without_nickname_no_stray_quotes():
    assert fg._format_fighter_name("Jon Jones", None) == "Jon Jones"
    assert fg._format_fighter_name("Jon Jones", "") == "Jon Jones"
    assert fg._format_fighter_name("Jon Jones", "   ") == "Jon Jones"


def test_format_fighter_name_single_word_name_with_nickname():
    assert fg._format_fighter_name("Shogun", "The Axe Murderer") == 'Shogun "The Axe Murderer"'


def test_fighter_dropdown_label_no_disambiguator_is_plain_name():
    f = {"fighter_id": "x", "fighter_name": "Jon Jones", "nick_name": "Bones", "display_name": "Jon Jones"}
    # Nickname appended after the full name (not inserted between first and
    # last) so "Jon Jones" stays a contiguous, searchable substring of the
    # label - see _fighter_dropdown_label's docstring.
    assert fg._fighter_dropdown_label(f) == 'Jon Jones "Bones"'


def test_fighter_dropdown_label_carries_over_dob_suffix():
    f = {"fighter_id": "x", "fighter_name": "Michael McDonald", "nick_name": None,
         "display_name": "Michael McDonald (b. 1991-01-15)"}
    assert fg._fighter_dropdown_label(f) == "Michael McDonald (b. 1991-01-15)"


def test_fighter_dropdown_label_carries_over_fighter_id_suffix():
    # api/main.py's _build_fighter_index appends a "[id:...]" suffix when
    # two distinct fighter_ids share a name AND have no DOB to tell them
    # apart (see test_api.py's fighter-identity-integrity tests) - this
    # picker label must surface that disambiguator too, not just the
    # dob-shaped one, or the dropdown would still show two
    # indistinguishable "Same Name" entries despite the API already having
    # fixed the ambiguity.
    f = {"fighter_id": "daff32bc", "fighter_name": "Anthony Figueroa", "nick_name": None,
         "display_name": "Anthony Figueroa [id:daff32bc]"}
    assert fg._fighter_dropdown_label(f) == "Anthony Figueroa [id:daff32bc]"


def test_fighter_dropdown_label_carries_over_combined_dob_and_id_suffix():
    f = {"fighter_id": "x", "fighter_name": "Same Name", "nick_name": "Nick",
         "display_name": "Same Name (b. 1990-01-01) [id:abcd1234]"}
    assert fg._fighter_dropdown_label(f) == 'Same Name "Nick" (b. 1990-01-01) [id:abcd1234]'


# Real data: data/fighter.csv has Sean O'Malley and Rashad Evans both with
# nick_name "Suga" - the user's example query is "Sugar", not "Suga", so
# matching has to be bidirectional-substring, not a literal substring check.
def test_nickname_search_sugar_matches_sean_omalley():
    assert fg._fighter_matches_query("Sean O'Malley", "Suga", "Sugar") is True


def test_nickname_search_sugar_matches_rashad_evans():
    assert fg._fighter_matches_query("Rashad Evans", "Suga", "Sugar") is True


def test_nickname_search_is_case_insensitive():
    assert fg._fighter_matches_query("Sean O'Malley", "Suga", "sugar") is True
    assert fg._fighter_matches_query("Sean O'Malley", "SUGA", "sugar") is True


def test_search_matches_first_name():
    assert fg._fighter_matches_query("Jon Jones", None, "jon") is True


def test_search_matches_last_name():
    assert fg._fighter_matches_query("Jon Jones", None, "jones") is True


def test_search_matches_full_name_without_needing_the_nickname():
    # The reported bug: searching the fighter's real full name should work
    # without typing their nickname in between (e.g. "Jon Bones Jones").
    assert fg._fighter_matches_query("Jon Jones", "Bones", "Jon Jones") is True
    assert fg._fighter_matches_query("Jon Jones", "Bones", "jon jones") is True


def test_search_full_name_matches_in_the_actual_dropdown_label_too():
    # Guards the pairing between _fighter_matches_query and
    # _fighter_dropdown_label: Gradio's own live filter matches against the
    # rendered label text, so "Jon Jones" must be a literal substring of it
    # even though the fighter has a nickname.
    f = {"fighter_id": "x", "fighter_name": "Jon Jones", "nick_name": "Bones", "display_name": "Jon Jones"}
    assert "jon jones" in fg._fighter_dropdown_label(f).lower()


def test_search_with_missing_nickname_does_not_crash_and_does_not_match_unrelated_query():
    assert fg._fighter_matches_query("Jon Jones", None, "bones") is False


def test_search_empty_query_matches_everything():
    assert fg._fighter_matches_query("Jon Jones", None, "") is True
    assert fg._fighter_matches_query("Jon Jones", None, None) is True


def test_search_no_match_returns_false():
    assert fg._fighter_matches_query("Jon Jones", "Bones", "khabib") is False


def test_filter_fighters_narrows_list_by_nickname():
    fighters = [
        {"fighter_id": "1", "fighter_name": "Sean O'Malley", "nick_name": "Suga"},
        {"fighter_id": "2", "fighter_name": "Jon Jones", "nick_name": "Bones"},
        {"fighter_id": "3", "fighter_name": "Rashad Evans", "nick_name": "Suga"},
    ]
    result = fg._filter_fighters(fighters, "Sugar")
    assert {f["fighter_id"] for f in result} == {"1", "3"}


def test_filter_fighters_empty_query_returns_all():
    fighters = [{"fighter_id": "1", "fighter_name": "Jon Jones", "nick_name": None}]
    assert fg._filter_fighters(fighters, "") == fighters


def _make_actual_fight_stats(sig_str_landed=10, td_success=1):
    return {
        "sig_str_landed": sig_str_landed, "sig_str_atmp": sig_str_landed + 5,
        "td_success": td_success, "td_atmp": td_success + 1, "sub_att": 0,
        "sig_str_landed_absorbed": 8, "sig_str_atmp_absorbed": 15,
        "won": False, "method": "Overturned", "finish_round": 3, "finish_time_seconds": 120,
    }


# ---------------------------------------------------------------------------
# Regression: real UFC 214 (Cormier vs. Jones 2, overturned to a no-contest)
# and other non-standard outcomes - the model's pre-fight Red/Blue
# prediction must be retained and shown separately from the actual
# official result, never suppressed and never forced into a fabricated
# winner. See api/main.py's _meeting_winner / tests/test_api.py's
# test_past_meeting_retains_predicted_winner_for_a_real_no_contest for the
# same case at the API layer.
# ---------------------------------------------------------------------------

def _make_predicted(winner="red", red_prob=0.952, predicted_round=None, predicted_method=None):
    return {
        "predicted_winner": winner, "red_win_probability": red_prob, "blue_win_probability": 1 - red_prob,
        "predicted_round": predicted_round, "predicted_method": predicted_method,
    }


def test_format_meeting_retains_predicted_winner_for_a_no_contest():
    meeting = {
        "event_name": "UFC 214: Cormier vs. Jones 2", "event_date": "2017-07-29",
        "predicted": _make_predicted("red", 0.952),
        "actual": {
            "winner": "no_contest", "result_status": "no_contest", "method": "Overturned",
            "details": "Failed Drug Test by Jones", "finish_round": 3, "finish_time_seconds": 181,
            "red": _make_actual_fight_stats(95, 0), "blue": _make_actual_fight_stats(58, 0),
        },
    }
    text = fg._format_meeting(meeting, "Jon Jones", "Daniel Cormier")

    # the pre-fight prediction is retained and shown, not suppressed.
    assert "Predicted" in text
    assert "95.2% red" in text
    assert "4.8% blue" in text
    # no round/win-method model exists yet - shown as Unavailable, never guessed.
    assert "Predicted round: Unavailable" in text
    assert "Predicted method: Unavailable" in text
    # never a fabricated agree/disagree against a non-existent winner.
    assert "model agreed" not in text
    assert "model disagreed" not in text
    # the actual result is a no-contest, not a forced Red/Blue winner.
    assert "**No Contest**" in text
    assert "Overturned" in text
    assert "Failed Drug Test by Jones" in text
    assert "Round 3, 3:01" in text
    # fight stats recorded before the stoppage are still shown.
    assert "95 sig. str landed" in text


def test_format_meeting_retains_predicted_winner_for_a_draw():
    meeting = {
        "event_name": "Some Event", "event_date": "2020-01-01",
        "predicted": _make_predicted("blue", 0.4),
        "actual": {
            "winner": "draw", "result_status": "draw", "method": "Decision - Majority",
            "details": None, "finish_round": 3, "finish_time_seconds": 300,
            "red": _make_actual_fight_stats(20, 1), "blue": _make_actual_fight_stats(18, 1),
        },
    }
    text = fg._format_meeting(meeting, "Red Fighter", "Blue Fighter")

    assert "Predicted" in text
    assert "40.0% red" in text
    assert "**Draw**" in text
    assert "model agreed" not in text
    assert "model disagreed" not in text


def test_format_meeting_dq_is_decisive_and_still_scored_agree_or_disagree():
    # A DQ has a real Red/Blue winner (unlike a draw/no-contest), so the
    # usual agree/disagree comparison against the prediction still applies,
    # and the actual winner's real name (not "Blue") is shown.
    meeting = {
        "event_name": "Some Event", "event_date": "2020-01-01",
        "predicted": _make_predicted("blue", 0.3),
        "actual": {
            "winner": "blue", "result_status": "win", "method": "DQ",
            "details": "Illegal Elbows by Red Fighter", "finish_round": 2, "finish_time_seconds": 60,
            "red": _make_actual_fight_stats(5, 0), "blue": _make_actual_fight_stats(10, 1),
        },
    }
    text = fg._format_meeting(meeting, "Red Fighter", "Blue Fighter")

    assert "✅ model agreed" in text
    assert "**Blue Fighter** won by DQ" in text
    assert "Round 2, 1:00" in text
    assert "Illegal Elbows by Red Fighter" in text


def test_format_meeting_shows_predicted_round_and_method_once_available():
    # Future-proofing: once the round/win-method targets are trained
    # (docs/tasks.md Recommendation), api/main.py populates these instead
    # of None - the frontend must show the real values, not "Unavailable".
    meeting = {
        "event_name": "Some Event", "event_date": "2020-01-01",
        "predicted": _make_predicted("red", 0.8, predicted_round=2, predicted_method="KO/TKO"),
        "actual": {
            "winner": "red", "result_status": "win", "method": "KO/TKO",
            "details": None, "finish_round": 2, "finish_time_seconds": 90,
            "red": _make_actual_fight_stats(30, 0), "blue": _make_actual_fight_stats(10, 0),
        },
    }
    text = fg._format_meeting(meeting, "Red Fighter", "Blue Fighter")
    assert "Predicted round: 2" in text
    assert "Predicted method: KO/TKO" in text
    assert "Unavailable" not in text


def test_format_meeting_no_details_omits_the_dash_note():
    meeting = {
        "event_name": "Some Event", "event_date": "2020-01-01",
        "predicted": None,
        "actual": {
            "winner": "red", "result_status": "win", "method": "Decision - Unanimous",
            "details": None, "finish_round": 3, "finish_time_seconds": 300,
            "red": _make_actual_fight_stats(20, 1), "blue": _make_actual_fight_stats(15, 0),
        },
    }
    text = fg._format_meeting(meeting, "Red Fighter", "Blue Fighter")
    assert "Predicted: Unavailable" in text
    assert "**Red Fighter** won by Decision - Unanimous" in text
    assert "Round 3, 5:00" in text


def test_format_seconds_formats_minutes_and_seconds():
    assert fg._format_seconds(0) == "0:00"
    assert fg._format_seconds(61) == "1:01"
    assert fg._format_seconds(181) == "3:01"


def test_format_seconds_none_stays_none_not_a_fabricated_zero():
    assert fg._format_seconds(None) is None


def _make_stats(prior_fights, prior_wins=0, prior_draws=0, prior_no_contests=0):
    # minimal build_display_stats-shaped dict - only the fields
    # _format_tale_of_the_tape/_low_sample_caveat actually read.
    prior_losses = prior_fights - prior_wins - prior_draws - prior_no_contests
    decided = prior_wins + prior_losses
    return {
        "striking": {"slpm": 1.0, "striking_accuracy": 0.5, "sapm": 1.0, "striking_defense": 0.5},
        "grappling": {"takedown_avg_per_15min": 0.0, "takedown_accuracy": None,
                      "takedown_defense": None, "sub_att_per_15min": 0.0},
        "physical": {"age": 30.0, "stance": "Orthodox", "height_inches": 70.0,
                     "reach_inches": 72.0, "strength_of_schedule": 0.5},
        "fight_pace": {"avg_fight_time_min": 10.0, "method_of_victory": {
            "occurrence": {"ko_tko": 0.0, "submission": 0.0, "decision": 0.0, "other": 0.0},
            "wins_by_method": {"ko_tko": 0.0, "submission": 0.0, "decision": 0.0, "other": 0.0}}},
        "record": {"prior_fights": prior_fights, "prior_wins": prior_wins,
                   "prior_losses": prior_losses, "prior_draws": prior_draws,
                   "prior_no_contests": prior_no_contests,
                   "prior_win_rate": (prior_wins / decided) if decided else None,
                   "win_streak": 0, "loss_streak": 0, "form_last5_win_rate": None},
    }


def test_low_sample_caveat_absent_when_both_corners_have_enough_fights():
    red = _make_stats(prior_fights=10, prior_wins=8)
    blue = _make_stats(prior_fights=20, prior_wins=15)
    assert fg._low_sample_caveat("Red Fighter", red, "Blue Fighter", blue) == ""


def test_low_sample_caveat_names_the_thin_corner_with_its_fight_count():
    red = _make_stats(prior_fights=1, prior_wins=1)
    blue = _make_stats(prior_fights=20, prior_wins=15)
    note = fg._low_sample_caveat("Gable Steveson", red, "Jon Jones", blue)
    assert "Gable Steveson (1 prior UFC fight)" in note
    assert "Jon Jones" not in note


def test_low_sample_caveat_names_both_corners_when_both_are_thin():
    red = _make_stats(prior_fights=0)
    blue = _make_stats(prior_fights=2, prior_wins=1)
    note = fg._low_sample_caveat("Red Fighter", red, "Blue Fighter", blue)
    assert "Red Fighter (0 prior UFC fights)" in note
    assert "Blue Fighter (2 prior UFC fights)" in note


def test_format_tale_of_the_tape_appends_low_sample_caveat():
    red = _make_stats(prior_fights=1, prior_wins=1)
    blue = _make_stats(prior_fights=20, prior_wins=15)
    tape = fg._format_tale_of_the_tape("Gable Steveson", red, "Jon Jones", blue)
    assert "Gable Steveson (1 prior UFC fight)" in tape


def test_format_tale_of_the_tape_no_caveat_when_both_corners_established():
    red = _make_stats(prior_fights=10, prior_wins=8)
    blue = _make_stats(prior_fights=20, prior_wins=15)
    tape = fg._format_tale_of_the_tape("Red Fighter", red, "Blue Fighter", blue)
    assert "⚠️" not in tape


def test_tale_of_the_tape_record_shows_w_l_d_nc_with_draws_and_no_contests_separate():
    # 10 fights: 6 wins, 1 draw, 1 no-contest -> 2 losses (10 - 6 - 1 - 1).
    # win_rate excludes the draw/NC from its denominator: 6 / (6 + 2) = 75%.
    red = _make_stats(prior_fights=10, prior_wins=6, prior_draws=1, prior_no_contests=1)
    blue = _make_stats(prior_fights=5, prior_wins=5)
    tape = fg._format_tale_of_the_tape("Red Fighter", red, "Blue Fighter", blue)
    assert "UFC Record (W-L-D-NC)" in tape
    assert "6-2-1-1 (75.0%)" in tape
    assert "5-0-0-0 (100.0%)" in tape


class _FakeResponse:
    """Stands in for a requests.Response - json_raises=True simulates a
    non-JSON body (a proxy error page, a plain-text 500/502), which is
    exactly what crashed predict()'s old unconditional resp.json() call in
    production."""

    def __init__(self, status_code=200, json_data=None, text="", json_raises=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)
        return self._json_data


# ---------------------------------------------------------------------------
# Issue 3: non-JSON and unsuccessful API responses must not raise a raw
# requests/JSON exception out of _get()/predict() - see _response_error_
# detail and _get.
# ---------------------------------------------------------------------------

def test_response_error_detail_prefers_json_detail_key():
    resp = _FakeResponse(status_code=400, json_data={"detail": "fighter not found"})
    assert fg._response_error_detail(resp) == "fighter not found"


def test_response_error_detail_falls_back_to_raw_text_when_not_json():
    resp = _FakeResponse(status_code=502, text="<html>Bad Gateway</html>", json_raises=True)
    assert fg._response_error_detail(resp) == "<html>Bad Gateway</html>"


def test_response_error_detail_blank_non_json_body_falls_back_to_status():
    resp = _FakeResponse(status_code=500, text="", json_raises=True)
    assert fg._response_error_detail(resp) == "HTTP 500"


def test_get_raises_gr_error_not_raw_exception_on_connection_failure(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("connection refused")
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(fg.gr.Error):
        fg._get("/models")


def test_get_raises_gr_error_on_non_200_status(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(
        status_code=500, json_raises=True, text="Internal Server Error"))
    with pytest.raises(fg.gr.Error):
        fg._get("/models")


def test_get_raises_gr_error_on_non_json_200_response(monkeypatch):
    # The real production trigger: the API said 200 but the body isn't
    # JSON at all - resp.json() must not be allowed to raise uncaught.
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(
        status_code=200, json_raises=True, text="not json"))
    with pytest.raises(fg.gr.Error):
        fg._get("/models")


def test_get_returns_json_on_success(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(
        status_code=200, json_data=[{"id": "m1"}]))
    assert fg._get("/models") == [{"id": "m1"}]


# ---------------------------------------------------------------------------
# Issue 1 (+4): dropdowns whose real choices only exist once a later event
# populates them must allow a stale/custom value through rather than have
# Gradio's own Dropdown.preprocess() hard-raise on it - see the comment
# above model_dropdown/division_dropdown/red_fighter/blue_fighter.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Issue 2: Tale of the Tape's two fighter columns (red/blue values either
# side of the label column) must render at equal width, not whatever width
# each column's own longest cell happens to need.
# ---------------------------------------------------------------------------

def test_tale_of_the_tape_panel_has_the_elem_id_the_width_css_targets():
    assert fg.tale_of_the_tape_panel.elem_id == "tale-of-the-tape"


def test_tale_of_the_tape_css_forces_equal_width_fighter_columns():
    css = fg.TALE_OF_THE_TAPE_CSS
    assert "#tale-of-the-tape" in css
    assert "table-layout: fixed" in css
    # first-child (red value column) and last-child (blue value column)
    # must be declared together in one rule, so they always share the
    # exact same width - not two separately-tunable widths that could
    # drift apart.
    import re
    rule = re.search(r"first-child.*?last-child[^{]*\{\s*width:\s*([\d.]+%)\s*;?\s*\}", css, re.S)
    assert rule is not None
    # And that shared width must differ from the (narrower) label column's
    # own width, confirming this rule is actually doing something.
    label_rule = re.search(r"nth-child\(2\)[^{]*\{\s*width:\s*([\d.]+%)", css)
    assert label_rule is not None and label_rule.group(1) != rule.group(1)


def test_dynamic_dropdowns_allow_custom_value_so_gradio_never_hard_raises():
    assert fg.model_dropdown.allow_custom_value is True
    assert fg.division_dropdown.allow_custom_value is True
    assert fg.red_fighter.allow_custom_value is True
    assert fg.blue_fighter.allow_custom_value is True


_FIGHTERS = [
    {"fighter_id": "r1", "fighter_name": "Red Fighter", "nick_name": None, "years": [2020, 2021]},
    {"fighter_id": "b1", "fighter_name": "Blue Fighter", "nick_name": None, "years": [2021]},
]


def test_on_fighter_change_invalid_selection_does_not_crash(monkeypatch):
    # The value a Dropdown.preprocess() would otherwise have rejected
    # outright (empty choices, or a value from a stale/other division)
    # now reaches this handler - it must fall back cleanly instead of
    # next()-with-no-default raising StopIteration.
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], []))
    year_update, other_fighter_update, other_year_update = fg.on_fighter_change("ping", "ping", None)
    assert year_update["choices"] == [fg.NO_YEAR_CHOICE]
    assert year_update["value"] == fg.NO_YEAR_CHOICE


def test_on_fighter_change_valid_selection_still_populates_years(monkeypatch):
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    year_update, other_fighter_update, other_year_update = fg.on_fighter_change("Lightweight", "r1", None)
    assert year_update["choices"] == [fg.NO_YEAR_CHOICE, "2020", "2021"]


def test_on_fighter_photo_change_valid_fighter_fetches_and_shows_photo(monkeypatch):
    monkeypatch.setattr(fg.images, "get_fighter_image_url", lambda fid, name: "https://ufc.com/x.png")
    results = list(fg.on_fighter_photo_change("r1", _FIGHTERS, False))
    _html, selected, _predict_update = results[-1]
    assert selected is True
    assert "<img" in _html


def test_on_fighter_photo_change_blank_fighter_id_shows_placeholder_without_fetch(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not fetch a photo when no fighter is selected")
    monkeypatch.setattr(fg.images, "get_fighter_image_url", _boom)

    results = list(fg.on_fighter_photo_change(None, _FIGHTERS, False))
    assert len(results) == 1
    html, selected, predict_update = results[0]
    assert selected is False
    assert predict_update["interactive"] is False
    assert "Select a fighter" in html


def test_on_fighter_photo_change_invalid_custom_value_never_fetches_or_caches_a_photo(monkeypatch):
    # Issue 3: the Fighter dropdown allows a custom typed value (see the
    # allow_custom_value comment), so this can fire with a fighter_id that
    # matches nothing in `fighters` - e.g. a random/invalid search string.
    # That must be treated like "no fighter selected", never as a real
    # fighter whose (fabricated) name gets looked up and potentially cached.
    def _boom(*a, **k):
        raise AssertionError("must not fetch/cache an image for an invalid selection")
    monkeypatch.setattr(fg.images, "get_fighter_image_url", _boom)

    results = list(fg.on_fighter_photo_change("some random gibberish", _FIGHTERS, False))
    assert len(results) == 1
    html, selected, predict_update = results[0]
    assert selected is False
    assert predict_update["interactive"] is False
    assert "Select a fighter" in html


def test_on_fighter_change_blank_fighter_id_short_circuits_without_a_lookup(monkeypatch):
    def _boom(division):
        raise AssertionError("must not call the API when no fighter is selected")
    monkeypatch.setattr(fg, "_fighter_choices", _boom)
    year_update, _other, _other_year = fg.on_fighter_change("Lightweight", None, None)
    assert year_update["value"] == fg.NO_YEAR_CHOICE


# ---------------------------------------------------------------------------
# Issue 2/3 combined, from predict()'s point of view: an invalid selection
# must be rejected before ever calling the API, and a failed/malformed API
# call must come back as a clean error string, never a crash.
# ---------------------------------------------------------------------------

def test_predict_rejects_fighter_id_not_in_current_choices_without_calling_api(monkeypatch):
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], []))

    def _boom(*a, **k):
        raise AssertionError("predict() must not call the API for an invalid selection")
    monkeypatch.setattr(requests, "post", _boom)

    headline, tape, history, last_result = fg.predict("m1", "ping", "ping", None, "ping", None)
    assert headline.startswith("Error:")
    assert last_result is None


def test_predict_handles_connection_error_gracefully(monkeypatch):
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")
    monkeypatch.setattr(requests, "post", _boom)

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert headline.startswith("Error:")


def test_predict_handles_non_json_error_response_gracefully(monkeypatch):
    # The exact reported crash: resp.json() on a non-JSON error body.
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(
        status_code=502, text="<html>Bad Gateway</html>", json_raises=True))

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert headline == "Error: <html>Bad Gateway</html>"


def test_predict_handles_non_json_200_response_gracefully(monkeypatch):
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(
        status_code=200, json_raises=True, text="not json"))

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert headline.startswith("Error:")


def test_predict_surfaces_json_detail_from_a_400_response(monkeypatch):
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(
        status_code=400, json_data={"detail": "fighter r1 has no fight history"}))

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert headline == "Error: fighter r1 has no fight history"


def test_predict_headline_shows_predicted_round_and_method_for_an_upcoming_fight(monkeypatch):
    # Not a past meeting - an upcoming/hypothetical matchup. The headline
    # built from /predict's own top-level response (not a past_meetings
    # entry) must show the predicted round/method too, right under the
    # predicted winner, the same way a past meeting already does.
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(
        status_code=200, json_data={
            "red_win_probability": 0.7, "blue_win_probability": 0.3, "predicted_winner": "red",
            "predicted_round": 3, "predicted_method": "Decision",
            "red_snapshot_date": "2021-01-01", "blue_snapshot_date": "2021-06-01",
            "red_stats": _make_stats(prior_fights=10, prior_wins=8),
            "blue_stats": _make_stats(prior_fights=10, prior_wins=6),
            "past_meetings": [],
        }))

    headline, _tape, history, _last_result = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert "Predicted winner: Red Fighter" in headline
    assert "Predicted round: 3" in headline
    assert "Predicted method: Decision" in headline
    assert history == "These two fighters have no completed fight on record against each other."


def test_predict_headline_shows_unavailable_when_round_and_method_are_missing(monkeypatch):
    # Mirrors _format_meeting's own "Unavailable" fallback (never a guessed
    # value) for an older API/model artifact that doesn't yet return these.
    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(
        status_code=200, json_data={
            "red_win_probability": 0.7, "blue_win_probability": 0.3, "predicted_winner": "red",
            "predicted_round": None, "predicted_method": None,
            "red_snapshot_date": "2021-01-01", "blue_snapshot_date": "2021-06-01",
            "red_stats": _make_stats(prior_fights=10, prior_wins=8),
            "blue_stats": _make_stats(prior_fights=10, prior_wins=6),
            "past_meetings": [],
        }))

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None)
    assert "Predicted round: Unavailable" in headline
    assert "Predicted method: Unavailable" in headline


# ---------------------------------------------------------------------------
# Fight Format toggle (3 vs. 5 rounds, see THREE_ROUNDS_CHOICE/
# FIVE_ROUNDS_CHOICE and fight_format_radio) - must actually change the
# rounds_scheduled the frontend sends to the API, not just the button's own
# label, and must display whatever format the API says it actually used.
# ---------------------------------------------------------------------------

def test_predict_sends_rounds_scheduled_3_for_the_default_three_rounds_choice(monkeypatch):
    captured = {}

    def _capture_post(url, json=None, **kwargs):
        captured.update(json)
        return _FakeResponse(status_code=200, json_data={
            "red_win_probability": 0.6, "blue_win_probability": 0.4, "predicted_winner": "red",
            "predicted_round": 3, "predicted_method": "Decision", "rounds_scheduled": 3,
            "red_snapshot_date": "2021-01-01", "blue_snapshot_date": "2021-06-01",
            "red_stats": _make_stats(prior_fights=10, prior_wins=8),
            "blue_stats": _make_stats(prior_fights=10, prior_wins=6),
            "past_meetings": [],
        })

    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", _capture_post)

    fg.predict("m1", "Lightweight", "r1", None, "b1", None, fg.THREE_ROUNDS_CHOICE)
    assert captured["rounds_scheduled"] == 3


def test_predict_sends_rounds_scheduled_5_for_the_five_rounds_choice(monkeypatch):
    captured = {}

    def _capture_post(url, json=None, **kwargs):
        captured.update(json)
        return _FakeResponse(status_code=200, json_data={
            "red_win_probability": 0.6, "blue_win_probability": 0.4, "predicted_winner": "red",
            "predicted_round": 4, "predicted_method": "Decision", "rounds_scheduled": 5,
            "red_snapshot_date": "2021-01-01", "blue_snapshot_date": "2021-06-01",
            "red_stats": _make_stats(prior_fights=10, prior_wins=8),
            "blue_stats": _make_stats(prior_fights=10, prior_wins=6),
            "past_meetings": [],
        })

    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", _capture_post)

    headline, *_rest = fg.predict("m1", "Lightweight", "r1", None, "b1", None, fg.FIVE_ROUNDS_CHOICE)
    assert captured["rounds_scheduled"] == 5
    # the headline reflects what the API says it actually used, not just
    # what was requested - see predict()'s rounds_scheduled/format_suffix.
    assert "5-round fight" in headline


def test_predict_defaults_to_3_rounds_for_an_invalid_fight_format_value(monkeypatch):
    # fight_format_radio has no allow_custom_value, so this shouldn't reach
    # predict() in the live app - but a stale/direct call must still fail
    # safe to the far more common format rather than crashing on a KeyError.
    captured = {}

    def _capture_post(url, json=None, **kwargs):
        captured.update(json)
        return _FakeResponse(status_code=200, json_data={
            "red_win_probability": 0.6, "blue_win_probability": 0.4, "predicted_winner": "red",
            "predicted_round": 3, "predicted_method": "Decision", "rounds_scheduled": 3,
            "red_snapshot_date": "2021-01-01", "blue_snapshot_date": "2021-06-01",
            "red_stats": _make_stats(prior_fights=10, prior_wins=8),
            "blue_stats": _make_stats(prior_fights=10, prior_wins=6),
            "past_meetings": [],
        })

    monkeypatch.setattr(fg, "_fighter_choices", lambda division: ([], _FIGHTERS))
    monkeypatch.setattr(requests, "post", _capture_post)

    fg.predict("m1", "Lightweight", "r1", None, "b1", None, "garbage")
    assert captured["rounds_scheduled"] == 3
