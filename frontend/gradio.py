"""Thin Gradio UI for the UFC Elo matchup predictor. Calls the FastAPI
service in api/main.py over HTTP - no model or feature-engineering logic
lives here (see docs/proposal.md: "the frontend should consume the API
rather than duplicating model logic"). Run the API first:

    uvicorn api.main:app --reload

then this file:

    python frontend/gradio.py
"""
import os

import gradio as gr
import requests

from frontend import images

API_URL = os.environ.get("UFC_ELO_API_URL", "http://127.0.0.1:8000")
NO_YEAR_CHOICE = "Most recent (full history)"

# Upper weight limit per division - the UFC's own numbers (not model data,
# so they live here rather than in src/features/engineering.py). kg values
# are the UFC's own published international limits, not a lbs*0.4536
# conversion (those round differently, e.g. Heavyweight is officially
# 120.2kg, not 265lbs's literal 120.2017...kg - close, but these are the
# numbers actually printed on UFC material).
DIVISION_WEIGHT_LIMITS_LBS = {
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
    "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 265,
}
DIVISION_WEIGHT_LIMITS_KG = {
    "Flyweight": 56.7, "Bantamweight": 61.2, "Featherweight": 65.8, "Lightweight": 70.3,
    "Welterweight": 77.1, "Middleweight": 83.9, "Light Heavyweight": 93.0, "Heavyweight": 120.2,
}


def _response_error_detail(resp):
    """The error message to show for a non-2xx (or otherwise unreadable)
    API response - tolerating a body that isn't JSON at all. A FastAPI
    HTTPException's own shape is {"detail": "..."}, but a dead upstream, a
    reverse-proxy timeout/error page, or any uncaught non-ValueError
    exception on the API side hands back plain text or HTML instead, and
    calling resp.json() on that unconditionally is what crashed predict()
    (frontend/gradio.py) in production."""
    try:
        payload = resp.json()
    except ValueError:  # requests.exceptions.JSONDecodeError subclasses ValueError
        return resp.text.strip() or f"HTTP {resp.status_code}"
    if isinstance(payload, dict) and "detail" in payload:
        return payload["detail"]
    return resp.text


def _get(path, **params):
    """GET {API_URL}{path}, never raising a raw requests/JSON exception -
    every failure mode (API unreachable, non-2xx status, a response body
    that isn't JSON) becomes a gr.Error instead, which Gradio shows as a
    clean toast rather than a 500 traceback or an unhandled crash."""
    try:
        resp = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    except requests.exceptions.RequestException as exc:
        raise gr.Error(f"Could not reach the prediction API at {API_URL}: {exc}") from exc
    if resp.status_code != 200:
        raise gr.Error(f"The prediction API returned an error: {_response_error_detail(resp)}")
    try:
        return resp.json()
    except ValueError as exc:
        raise gr.Error("The prediction API returned a response that wasn't valid JSON.") from exc


def _fighter_dropdown_label(f):
    """The picker's own label: fight-title format ('First "Nick" Last'),
    with the API's disambiguator suffix (from display_name) carried over
    unchanged for duplicate names - "(b. ...)" when a DOB is on record,
    plus a short "[id:...]" fighter_id suffix on top of that for the rarer
    case where even name+DOB isn't enough to tell two real fighters apart
    (see api/main.py's _build_fighter_index: two distinct fighter_ids can
    share both a name and a missing DOB, e.g. two real "Anthony Figueroa"
    records - without ANY disambiguator they'd be two visually-identical
    dropdown entries with no way to tell which is which before picking).
    Whatever api/main.py appends after fighter_name is taken verbatim here,
    rather than re-deriving it, so this never drifts out of sync with the
    API's own disambiguation logic. Deliberately NOT the raw display_name
    - embedding the nickname here is what makes Gradio's own built-in,
    instant, client-side Dropdown filter match nicknames too (confirmed: a
    query like "Sugar" now matches a fighter labeled 'Sean "Suga" O\\'Malley
    (b. ...)' since "suga" is itself a substring of "sugar" - see
    _fighter_matches_query for the exact rule this mirrors). No server
    round-trip per keystroke needed for that - a per-keystroke custom
    filter (an earlier version of this file used Dropdown.key_up) raced
    against Gradio's own client-side filter and made selecting a fighter
    feel laggy/inconsistent; this removes that race entirely by not
    needing a second filter at all."""
    name = _format_fighter_name(f.get("fighter_name") or f.get("fighter_id", ""), f.get("nick_name"))
    raw_name = f.get("fighter_name") or ""
    display_name = f.get("display_name") or ""
    suffix = display_name[len(raw_name):].strip() if display_name.startswith(raw_name) else ""
    return f"{name} {suffix}" if suffix else name


def _fighter_choices(division):
    fighters = _get("/fighters", division=division)
    # (label, value) pairs: dropdown shows _fighter_dropdown_label, carries
    # the fighter_id as the actual value.
    return [(_fighter_dropdown_label(f), f["fighter_id"]) for f in fighters], fighters


def _inches_to_cm(inches):
    return round(inches * 2.54)


def _format_height(inches, unit):
    """70 inches, "lbs" -> '5\\'10"' (feet+inches, standard imperial height
    notation - not bare inches). 70 inches, "kg" -> '178 cm'. None (missing
    data, never invented) -> "-"."""
    if inches is None:
        return "-"
    if unit == "kg":
        return f"{_inches_to_cm(inches)} cm"
    feet, remaining_inches = divmod(round(inches), 12)
    return f"{feet}'{remaining_inches}\""


def _format_reach(inches, unit):
    """74 inches, "lbs" -> '74"'. 74 inches, "kg" -> '188 cm'."""
    if inches is None:
        return "-"
    return f"{_inches_to_cm(inches)} cm" if unit == "kg" else f"{round(inches)}\""


def _division_label(division, unit):
    """"Welterweight" -> "Welterweight (170 lbs)" or "Welterweight (77.1 kg)" -
    looked up from the UFC's own published limits in each unit, not computed
    by converting one into the other. Divisions with no known weight limit
    (shouldn't happen for the 8 MAIN_DIVISIONS the API serves, but
    defensive) fall back to the bare division name."""
    limits = DIVISION_WEIGHT_LIMITS_LBS if unit == "lbs" else DIVISION_WEIGHT_LIMITS_KG
    weight = limits.get(division)
    if weight is None:
        return division
    return f"{division} ({weight} {unit})"


def _division_choices(divisions, unit):
    return [(_division_label(d, unit), d) for d in divisions]


# Defined in frontend/images.py (the one-directional dependency: this file
# imports images.py, images.py never imports this file - see split_name's
# docstring there for why that direction matters).
_split_name = images.split_name
_format_fighter_name = images.format_fighter_name


def _fighter_matches_query(full_name, nickname, query):
    """The matching rule fighter search is built around: first name, last
    name, or nickname, case-insensitive substring, either direction (e.g.
    query "Sugar" matches a fighter whose on-record nickname is "Suga",
    since "suga" is itself a substring of "sugar"). Empty query matches
    everything. _fighter_dropdown_label's embedded nickname is what makes
    Gradio's own built-in Dropdown filter actually enforce this rule live,
    with no server round-trip; this function (and _filter_fighters below)
    stay as the tested, explicit statement of that rule, and are what a
    future server-side search (e.g. an API ?search= param) would reuse."""
    query = (query or "").strip().lower()
    if not query:
        return True
    first, last = _split_name(full_name)
    if query in first.lower() or query in last.lower():
        return True
    nickname = (nickname or "").strip().lower()
    return bool(nickname) and (query in nickname or nickname in query)


def _filter_fighters(fighters, query):
    return [f for f in fighters if _fighter_matches_query(f.get("fighter_name", ""), f.get("nick_name"), query)]


def _clear_results():
    """Blanks the prior matchup's Tale of the Tape, headline, past-meetings
    history, and stashed unit-toggle state. Fired whenever a new set of
    fighters starts loading (a fighter pick in either corner, or a division
    switch) so a stale tape from the previous matchup never lingers on
    screen next to a fresh photo/selection."""
    return "", "", "", None


def on_division_change(division):
    choices, fighters = _fighter_choices(division)
    placeholder = images.fighter_placeholder_html()
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
        gr.update(choices=choices, value=None),
        gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
        fighters,  # fighters_state - backs both corners' Fighter dropdown search
        # Explicit reset (not just relying on the fighter dropdowns' own
        # .change() cascade) so a division switch always clears stale
        # photos/readiness from the previous division's selections.
        placeholder, False,
        placeholder, False,
        gr.update(interactive=False),
        *_clear_results(),
    )


def on_fighter_change(division, fighter_id, other_fighter_id):
    """Refresh this corner's year choices for the newly picked fighter. If
    that fighter is already selected in the OTHER corner, a fighter can't
    face themselves - clear the other corner's fighter (and its year
    choices) instead of leaving two identical corners selected."""
    if not fighter_id:
        return gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE), gr.update(), gr.update()

    _, fighters = _fighter_choices(division)
    info = next((f for f in fighters if f["fighter_id"] == fighter_id), None)
    if info is None:
        # A stale or invalid selection - the division changed out from
        # under this pick, or (the Fighter dropdowns allow a custom value
        # so a direct API call can submit anything) fighter_id was never a
        # real choice for this division at all. Nothing to refresh years
        # from; fail the same clean way as "no fighter selected" instead
        # of crashing on a next() with no match and no default.
        return gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE), gr.update(), gr.update()

    year_update = gr.update(choices=[NO_YEAR_CHOICE] + [str(y) for y in info["years"]], value=NO_YEAR_CHOICE)

    if fighter_id == other_fighter_id:
        return (
            year_update,
            gr.update(value=None),
            gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
        )
    return year_update, gr.update(), gr.update()


def on_fighter_photo_change(fighter_id, fighters, other_corner_selected):
    """Drives one corner's photo through placeholder -> loading -> resolved
    (real ufc.com photo or an initials avatar - see frontend/images.py) and
    this corner's "selected" state. Predict only needs both corners
    SELECTED, not their photos resolved - it enables as soon as both
    fighters are picked, so clicking Predict and the photo fetch run
    concurrently rather than Predict waiting on the network call. A
    generator: Gradio streams each yield as its own UI update, so the
    loading state is genuinely shown while the photo fetch is in flight."""
    if not fighter_id:
        yield images.fighter_placeholder_html(), False, gr.update(interactive=False)
        return

    info = next((f for f in fighters if f["fighter_id"] == fighter_id), {})
    name = info.get("fighter_name", fighter_id)
    nickname = info.get("nick_name")
    predict_interactive = gr.update(interactive=bool(other_corner_selected))

    yield images.fighter_loading_html(name, nickname), True, predict_interactive

    image_url = images.get_fighter_image_url(fighter_id, name)
    yield images.fighter_photo_html(name, nickname, image_url), True, predict_interactive


# Below this many prior UFC fights, every per-minute/share/accuracy stat in
# the tape is a ratio built from very little data (e.g. one short fight can
# produce a 16+ sig. strikes/min pace that is arithmetically correct but
# not a stable career rate - see docs/results.md Known Limitations). Flagged
# here, display-only: the underlying numbers are unchanged, and the model
# already has prior_fights/experience_diff available to weigh this itself.
LOW_SAMPLE_FIGHT_THRESHOLD = 5


def _low_sample_caveat(red_name, red_stats, blue_name, blue_stats):
    """A one-line caveat naming any corner whose career-rate stats above are
    built from fewer than LOW_SAMPLE_FIGHT_THRESHOLD prior fights - empty
    string if neither corner is that new. Record counts (not rates) are
    exact regardless of sample size, so this only warns about the ratio
    stats (striking/grappling rates, strength of schedule)."""
    thin = [
        (name, stats["record"]["prior_fights"])
        for name, stats in ((red_name, red_stats), (blue_name, blue_stats))
        if stats["record"]["prior_fights"] < LOW_SAMPLE_FIGHT_THRESHOLD
    ]
    if not thin:
        return ""
    names = " and ".join(
        f"{name} ({int(n)} prior UFC fight{'s' if n != 1 else ''})" for name, n in thin)
    return (
        f"\n\n*⚠️ {names} - the rate-based stats above (striking/grappling "
        f"per-minute or per-15-min figures, accuracy, strength of schedule) "
        f"are built from very little fight data and may not be reliable "
        f"career indicators.*"
    )


def _format_tale_of_the_tape(red_name, red_stats, blue_name, blue_stats, unit="lbs"):
    def pct(x):
        return "-" if x is None else f"{x * 100:.1f}%"

    def num(x, digits=2):
        return "-" if x is None else f"{x:.{digits}f}"

    def record(stats):
        r = stats["record"]
        return f"{r['prior_wins']}-{r['prior_fights'] - r['prior_wins']} ({pct(r['prior_win_rate'])})"

    def streak(stats):
        r = stats["record"]
        return f"W{int(r['win_streak'])}" if r["win_streak"] else f"L{int(r['loss_streak'])}"

    def method_pct(stats, key):
        return pct(stats["fight_pace"]["method_of_victory"]["wins_by_method"][key])

    rows = [
        ("UFC Record", record(red_stats), record(blue_stats)),
        ("Current Streak", streak(red_stats), streak(blue_stats)),
        ("Age", num(red_stats["physical"]["age"], 1), num(blue_stats["physical"]["age"], 1)),
        ("Height", _format_height(red_stats["physical"]["height_inches"], unit),
         _format_height(blue_stats["physical"]["height_inches"], unit)),
        ("Reach", _format_reach(red_stats["physical"]["reach_inches"], unit),
         _format_reach(blue_stats["physical"]["reach_inches"], unit)),
        ("Stance", red_stats["physical"]["stance"], blue_stats["physical"]["stance"]),
        ("Strength of Schedule", pct(red_stats["physical"]["strength_of_schedule"]),
         pct(blue_stats["physical"]["strength_of_schedule"])),
        ("Sig. Strikes Landed/Min", num(red_stats["striking"]["slpm"]), num(blue_stats["striking"]["slpm"])),
        ("Striking Accuracy", pct(red_stats["striking"]["striking_accuracy"]),
         pct(blue_stats["striking"]["striking_accuracy"])),
        ("Sig. Strikes Absorbed/Min", num(red_stats["striking"]["sapm"]), num(blue_stats["striking"]["sapm"])),
        ("Striking Defense", pct(red_stats["striking"]["striking_defense"]),
         pct(blue_stats["striking"]["striking_defense"])),
        ("Takedowns/15min", num(red_stats["grappling"]["takedown_avg_per_15min"]),
         num(blue_stats["grappling"]["takedown_avg_per_15min"])),
        ("Takedown Accuracy", pct(red_stats["grappling"]["takedown_accuracy"]),
         pct(blue_stats["grappling"]["takedown_accuracy"])),
        ("Takedown Defense", pct(red_stats["grappling"]["takedown_defense"]),
         pct(blue_stats["grappling"]["takedown_defense"])),
        ("Sub. Attempts/15min", num(red_stats["grappling"]["sub_att_per_15min"]),
         num(blue_stats["grappling"]["sub_att_per_15min"])),
        ("Avg Fight Time", f"{num(red_stats['fight_pace']['avg_fight_time_min'], 1)} min",
         f"{num(blue_stats['fight_pace']['avg_fight_time_min'], 1)} min"),
        ("Wins by KO/TKO", method_pct(red_stats, "ko_tko"), method_pct(blue_stats, "ko_tko")),
        ("Wins by Submission", method_pct(red_stats, "submission"), method_pct(blue_stats, "submission")),
        ("Wins by Decision", method_pct(red_stats, "decision"), method_pct(blue_stats, "decision")),
    ]

    lines = ["### Tale of the Tape", "", f"| {red_name} | | {blue_name} |", "|---:|:---:|:---|"]
    lines += [f"| {red_val} | {label} | {blue_val} |" for label, red_val, blue_val in rows]
    return "\n".join(lines) + _low_sample_caveat(red_name, red_stats, blue_name, blue_stats)


def _format_meeting(meeting):
    predicted = meeting["predicted"]
    actual = meeting["actual"]
    winner_label = "Red" if actual["winner"] == "red" else "Blue"

    if predicted is None:
        prediction_line = (
            "Predicted (as of right before this fight): unavailable - one corner had no "
            "fight history yet (likely their promotional debut).\n\n"
        )
    else:
        predicted_label = "Red" if predicted["predicted_winner"] == "red" else "Blue"
        agree = "✅ model agreed" if predicted["predicted_winner"] == actual["winner"] else "❌ model disagreed"
        prediction_line = (
            f"Predicted (as of right before this fight): **{predicted_label}** to win "
            f"({predicted['red_win_probability'] * 100:.1f}% red / "
            f"{predicted['blue_win_probability'] * 100:.1f}% blue) - {agree}\n\n"
        )

    return (
        f"### {meeting['event_name']} ({meeting['event_date']})\n"
        f"{prediction_line}"
        f"Actual: **{winner_label}** won by {actual['method']} (round {actual['finish_round']})\n\n"
        f"Actual fight stats - Red: {actual['red']['sig_str_landed']} sig. str landed, "
        f"{actual['red']['td_success']} TD | Blue: {actual['blue']['sig_str_landed']} sig. str landed, "
        f"{actual['blue']['td_success']} TD"
    )


def predict(model_name, division, red_fighter_id, red_year, blue_fighter_id, blue_year, unit="lbs"):
    if not red_fighter_id or not blue_fighter_id:
        return "Pick both a red and a blue corner fighter first.", "", "", None

    _, fighters = _fighter_choices(division)
    info_by_id = {f["fighter_id"]: f for f in fighters}
    if red_fighter_id not in info_by_id or blue_fighter_id not in info_by_id:
        # A stale or invalid selection - e.g. the division changed out
        # from under one corner's pick, or (the Fighter dropdowns allow a
        # custom value, so a direct API call can submit anything) an id
        # that was never a real choice for this division at all. Nothing
        # valid to predict from; fail the same clean way as "pick both
        # fighters" rather than sending it to the API at all.
        return "Error: one or both selected fighters aren't valid for this division - pick again.", "", "", None
    red_info = info_by_id[red_fighter_id]
    blue_info = info_by_id[blue_fighter_id]
    # Fight-title format ('First "Nickname" Last", plain "First Last" with
    # no nickname on record) everywhere a fighter is named in a result -
    # the picker dropdown keeps its own display_name (with DOB) separately,
    # since that's a disambiguator, not a fight title.
    red_name = _format_fighter_name(red_info.get("fighter_name", red_fighter_id), red_info.get("nick_name"))
    blue_name = _format_fighter_name(blue_info.get("fighter_name", blue_fighter_id), blue_info.get("nick_name"))

    payload = {
        "model_name": model_name,
        "division": division,
        "red_fighter_id": red_fighter_id,
        "red_year": None if red_year in (None, NO_YEAR_CHOICE) else int(red_year),
        "blue_fighter_id": blue_fighter_id,
        "blue_year": None if blue_year in (None, NO_YEAR_CHOICE) else int(blue_year),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
    except requests.exceptions.RequestException as exc:
        return f"Error: could not reach the prediction API ({exc}).", "", "", None

    if resp.status_code != 200:
        return f"Error: {_response_error_detail(resp)}", "", "", None

    try:
        data = resp.json()
    except ValueError:
        return "Error: the prediction API returned a response that wasn't valid JSON.", "", "", None
    winner_name = red_name if data["predicted_winner"] == "red" else blue_name
    headline = (
        f"## Predicted winner: {winner_name}\n"
        f"{red_name} win probability: **{data['red_win_probability'] * 100:.1f}%** "
        f"(Most recent fight: {data['red_snapshot_date']})\n\n"
        f"{blue_name} win probability: **{data['blue_win_probability'] * 100:.1f}%** "
        f"(Most recent fight: {data['blue_snapshot_date']})\n\n"
        f"Model used: {model_name}"
    )

    tale_of_the_tape = _format_tale_of_the_tape(red_name, data["red_stats"], blue_name, data["blue_stats"], unit)

    if data["past_meetings"]:
        history = "\n\n---\n\n".join(_format_meeting(m) for m in data["past_meetings"])
    else:
        history = "These two fighters have no completed fight on record against each other."

    # Stashed so the unit toggle can re-render the Tale of the Tape in the
    # new unit instantly, without a new /predict call - see on_unit_toggle.
    last_result = {"red_name": red_name, "red_stats": data["red_stats"],
                    "blue_name": blue_name, "blue_stats": data["blue_stats"]}
    return headline, tale_of_the_tape, history, last_result


with gr.Blocks(title="UFC Matchup Predictor") as demo:
    unit_state = gr.State("lbs")
    divisions_state = gr.State([])  # raw division names, for relabeling on unit toggle
    fighters_state = gr.State([])  # raw fighter dicts for the current division, for search
    last_result_state = gr.State(None)  # last predict() result, for an instant unit-toggle re-render
    # Whether each corner currently has a fighter picked - NOT whether their
    # photo has finished loading (Predict only waits on selection, so it and
    # the photo fetch run concurrently - see on_fighter_photo_change).
    red_selected_state = gr.State(False)
    blue_selected_state = gr.State(False)

    with gr.Row():
        gr.Markdown("# UFC Matchup Predictor", scale=5)
        unit_toggle_button = gr.Button("Switch to kg/cm", scale=1, size="sm")
    gr.Markdown(
        "Pick a weight division, then a fighter for each corner (type a name or nickname to "
        "narrow the Fighter list). The optional year picks that fighter's own latest fight "
        "within that year as the data cutoff - leave it blank to use their full history. "
        "Corner color has no effect on the prediction."
    )

    # allow_custom_value=True on every dropdown whose real choices only
    # exist once on_load()/on_division_change() populates them: without it,
    # Gradio's own Dropdown.preprocess() hard-raises ("Value: 'x' is not in
    # the list of choices: []") for ANY request that reaches this Blocks
    # app before that happens (a fresh session hitting the queue API
    # before its load event completes, a stale browser tab replaying an
    # old selection, or a direct API call) - and that raise, deep in
    # Gradio's queue/SSE handling rather than our own code, is what was
    # cascading into the "response already started" crash. The handler
    # functions below (on_fighter_change, predict) are the actual guard
    # now - they treat a value that isn't one of the current choices as a
    # stale/invalid selection and fail gracefully instead of crashing.
    # filterable=True here (previously False - only 3 choices) because
    # Gradio requires it whenever allow_custom_value=True; harmless given
    # how few choices this dropdown ever has.
    model_dropdown = gr.Dropdown(label="Model", choices=[], filterable=True, allow_custom_value=True)
    division_dropdown = gr.Dropdown(label="Weight Division", choices=[], filterable=True,
                                     allow_custom_value=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Red Corner")
            red_fighter = gr.Dropdown(label="Fighter (type a name or nickname to search)",
                                       choices=[], filterable=True, allow_custom_value=True)
            red_year = gr.Dropdown(label="Year (optional)", choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE)
        with gr.Column():
            gr.Markdown("### Blue Corner")
            blue_fighter = gr.Dropdown(label="Fighter (type a name or nickname to search)",
                                        choices=[], filterable=True, allow_custom_value=True)
            blue_year = gr.Dropdown(label="Year (optional)", choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE)

    # Disabled until both corners have a fighter picked AND that fighter's
    # photo lookup has finished - see on_fighter_photo_change.
    predict_button = gr.Button("Predict", variant="primary", interactive=False)

    result_headline = gr.Markdown()
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=160):
            red_photo = gr.HTML(images.fighter_placeholder_html())
        with gr.Column(scale=3):
            tale_of_the_tape_panel = gr.Markdown(label="Tale of the Tape")
        with gr.Column(scale=1, min_width=160):
            blue_photo = gr.HTML(images.fighter_placeholder_html())
    rematch_history = gr.Markdown(label="Past Meetings")

    division_dropdown.change(
        on_division_change, inputs=[division_dropdown],
        outputs=[red_fighter, red_year, blue_fighter, blue_year, fighters_state,
                  red_photo, red_selected_state, blue_photo, blue_selected_state, predict_button,
                  result_headline, tale_of_the_tape_panel, rematch_history, last_result_state])
    # A fighter can't face themselves: picking the same fighter already
    # selected in the other corner clears that other corner instead of
    # leaving two identical corners selected - see on_fighter_change.
    red_fighter.change(
        on_fighter_change, inputs=[division_dropdown, red_fighter, blue_fighter],
        outputs=[red_year, blue_fighter, blue_year])
    blue_fighter.change(
        on_fighter_change, inputs=[division_dropdown, blue_fighter, red_fighter],
        outputs=[blue_year, red_fighter, red_year])
    # Search is merged into the Fighter dropdown itself - no separate search
    # box, and no custom per-keystroke server handler either: the dropdown's
    # own label embeds the nickname (see _fighter_dropdown_label), so
    # Gradio's built-in client-side filter matches it instantly with zero
    # round-trips, and with no race against a server-pushed update.
    # Independent listener (alongside on_fighter_change above) driving this
    # corner's photo and the Predict button's enabled state - see
    # on_fighter_photo_change.
    red_fighter.change(
        on_fighter_photo_change, inputs=[red_fighter, fighters_state, blue_selected_state],
        outputs=[red_photo, red_selected_state, predict_button])
    blue_fighter.change(
        on_fighter_photo_change, inputs=[blue_fighter, fighters_state, red_selected_state],
        outputs=[blue_photo, blue_selected_state, predict_button])
    # A new fighter pick in either corner means a new matchup is loading -
    # clear the previous matchup's Tale of the Tape/headline/history so it
    # never sits stale next to the new selection (see _clear_results).
    red_fighter.change(
        _clear_results, outputs=[result_headline, tale_of_the_tape_panel, rematch_history, last_result_state])
    blue_fighter.change(
        _clear_results, outputs=[result_headline, tale_of_the_tape_panel, rematch_history, last_result_state])

    def on_unit_toggle(unit, divisions, current_division, last_result):
        new_unit = "kg" if unit == "lbs" else "lbs"
        button_label = "Switch to lbs/ft-in" if new_unit == "kg" else "Switch to kg/cm"
        # Re-render the already-shown Tale of the Tape in the new unit
        # immediately, from the stashed last result - no new /predict call,
        # and a no-op (gr.update() touches nothing) before any prediction
        # has been made yet.
        tale_update = gr.update()
        if last_result is not None:
            tale_update = gr.update(value=_format_tale_of_the_tape(
                last_result["red_name"], last_result["red_stats"],
                last_result["blue_name"], last_result["blue_stats"], new_unit))
        return (
            new_unit,
            gr.update(choices=_division_choices(divisions, new_unit), value=current_division),
            gr.update(value=button_label),
            tale_update,
        )

    unit_toggle_button.click(
        on_unit_toggle, inputs=[unit_state, divisions_state, division_dropdown, last_result_state],
        outputs=[unit_state, division_dropdown, unit_toggle_button, tale_of_the_tape_panel])

    predict_button.click(
        predict,
        inputs=[model_dropdown, division_dropdown, red_fighter, red_year, blue_fighter, blue_year, unit_state],
        outputs=[result_headline, tale_of_the_tape_panel, rematch_history, last_result_state],
    )

    @demo.load(outputs=[model_dropdown, division_dropdown, divisions_state])
    def on_load():
        models = _get("/models")
        divisions = _get("/divisions")
        return (
            gr.update(choices=[(m["label"], m["id"]) for m in models], value=models[0]["id"]),
            gr.update(choices=_division_choices(divisions, "lbs"), value=divisions[0]),
            divisions,
        )


if __name__ == "__main__":
    demo.launch()
