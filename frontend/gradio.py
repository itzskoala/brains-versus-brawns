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

API_URL = os.environ.get("UFC_ELO_API_URL", "http://127.0.0.1:8000")
NO_YEAR_CHOICE = "Most recent (full history)"


def _get(path, **params):
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fighter_choices(division):
    fighters = _get("/fighters", division=division)
    # (label, value) pairs: dropdown shows the display_name, carries the
    # fighter_id as the actual value - duplicate names are disambiguated by
    # display_name already including a date of birth (see api/main.py).
    return [(f["display_name"], f["fighter_id"]) for f in fighters], fighters


def on_division_change(division):
    choices, _ = _fighter_choices(division)
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
        gr.update(choices=choices, value=None),
        gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
    )


def on_fighter_change(division, fighter_id, other_fighter_id):
    """Refresh this corner's year choices for the newly picked fighter. If
    that fighter is already selected in the OTHER corner, a fighter can't
    face themselves - clear the other corner's fighter (and its year
    choices) instead of leaving two identical corners selected."""
    if not fighter_id:
        return gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE), gr.update(), gr.update()

    _, fighters = _fighter_choices(division)
    years = next(f["years"] for f in fighters if f["fighter_id"] == fighter_id)
    year_update = gr.update(choices=[NO_YEAR_CHOICE] + [str(y) for y in years], value=NO_YEAR_CHOICE)

    if fighter_id == other_fighter_id:
        return (
            year_update,
            gr.update(value=None),
            gr.update(choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE),
        )
    return year_update, gr.update(), gr.update()


def _format_tale_of_the_tape(red_name, red_stats, blue_name, blue_stats):
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
        ("Record", record(red_stats), record(blue_stats)),
        ("Current Streak", streak(red_stats), streak(blue_stats)),
        ("Age", num(red_stats["physical"]["age"], 1), num(blue_stats["physical"]["age"], 1)),
        ("Height", f"{num(red_stats['physical']['height_inches'], 0)}\"",
         f"{num(blue_stats['physical']['height_inches'], 0)}\""),
        ("Reach", f"{num(red_stats['physical']['reach_inches'], 0)}\"",
         f"{num(blue_stats['physical']['reach_inches'], 0)}\""),
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
    return "\n".join(lines)


def _format_meeting(meeting):
    predicted = meeting["predicted"]
    actual = meeting["actual"]
    winner_label = "Red" if actual["winner"] == "red" else "Blue"
    predicted_label = "Red" if predicted["predicted_winner"] == "red" else "Blue"
    agree = "✅ model agreed" if predicted["predicted_winner"] == actual["winner"] else "❌ model disagreed"

    return (
        f"### {meeting['event_name']} ({meeting['event_date']})\n"
        f"Predicted (as of right before this fight): **{predicted_label}** to win "
        f"({predicted['red_win_probability'] * 100:.1f}% red / "
        f"{predicted['blue_win_probability'] * 100:.1f}% blue) - {agree}\n\n"
        f"Actual: **{winner_label}** won by {actual['method']} (round {actual['finish_round']})\n\n"
        f"Actual fight stats - Red: {actual['red']['sig_str_landed']} sig. str landed, "
        f"{actual['red']['td_success']} TD | Blue: {actual['blue']['sig_str_landed']} sig. str landed, "
        f"{actual['blue']['td_success']} TD"
    )


def predict(model_name, division, red_fighter_id, red_year, blue_fighter_id, blue_year):
    if not red_fighter_id or not blue_fighter_id:
        return "Pick both a red and a blue corner fighter first.", "", ""

    _, fighters = _fighter_choices(division)
    name_by_id = {f["fighter_id"]: f["display_name"] for f in fighters}
    red_name = name_by_id.get(red_fighter_id, red_fighter_id)
    blue_name = name_by_id.get(blue_fighter_id, blue_fighter_id)

    payload = {
        "model_name": model_name,
        "division": division,
        "red_fighter_id": red_fighter_id,
        "red_year": None if red_year in (None, NO_YEAR_CHOICE) else int(red_year),
        "blue_fighter_id": blue_fighter_id,
        "blue_year": None if blue_year in (None, NO_YEAR_CHOICE) else int(blue_year),
    }
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
    if resp.status_code != 200:
        return f"Error: {resp.json().get('detail', resp.text)}", "", ""

    data = resp.json()
    winner_name = red_name if data["predicted_winner"] == "red" else blue_name
    headline = (
        f"## Predicted winner: {winner_name}\n"
        f"{red_name} win probability: **{data['red_win_probability'] * 100:.1f}%** "
        f"(Most recent fight: {data['red_snapshot_date']})\n\n"
        f"{blue_name} win probability: **{data['blue_win_probability'] * 100:.1f}%** "
        f"(Most recent fight: {data['blue_snapshot_date']})\n\n"
        f"Model used: {model_name}"
    )

    tale_of_the_tape = _format_tale_of_the_tape(red_name, data["red_stats"], blue_name, data["blue_stats"])

    if data["past_meetings"]:
        history = "\n\n---\n\n".join(_format_meeting(m) for m in data["past_meetings"])
    else:
        history = "These two fighters have no completed fight on record against each other."

    return headline, tale_of_the_tape, history


with gr.Blocks(title="UFC Matchup Predictor") as demo:
    gr.Markdown("# UFC Matchup Predictor")
    gr.Markdown(
        "Pick a weight division, then a fighter for each corner. The optional year picks that "
        "fighter's own latest fight within that year as the data cutoff - leave it blank to use "
        "their full history. Corner color has no effect on the prediction, it's just a label."
    )

    model_dropdown = gr.Dropdown(label="Model", choices=[], filterable=False)
    division_dropdown = gr.Dropdown(label="Weight Division", choices=[], filterable=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Red Corner")
            red_fighter = gr.Dropdown(label="Fighter", choices=[], filterable=True)
            red_year = gr.Dropdown(label="Year (optional)", choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE)
        with gr.Column():
            gr.Markdown("### Blue Corner")
            blue_fighter = gr.Dropdown(label="Fighter", choices=[], filterable=True)
            blue_year = gr.Dropdown(label="Year (optional)", choices=[NO_YEAR_CHOICE], value=NO_YEAR_CHOICE)

    predict_button = gr.Button("Predict", variant="primary")

    result_headline = gr.Markdown()
    tale_of_the_tape_panel = gr.Markdown(label="Tale of the Tape")
    rematch_history = gr.Markdown(label="Past Meetings")

    division_dropdown.change(
        on_division_change, inputs=[division_dropdown],
        outputs=[red_fighter, red_year, blue_fighter, blue_year])
    # A fighter can't face themselves: picking the same fighter already
    # selected in the other corner clears that other corner instead of
    # leaving two identical corners selected - see on_fighter_change.
    red_fighter.change(
        on_fighter_change, inputs=[division_dropdown, red_fighter, blue_fighter],
        outputs=[red_year, blue_fighter, blue_year])
    blue_fighter.change(
        on_fighter_change, inputs=[division_dropdown, blue_fighter, red_fighter],
        outputs=[blue_year, red_fighter, red_year])

    predict_button.click(
        predict,
        inputs=[model_dropdown, division_dropdown, red_fighter, red_year, blue_fighter, blue_year],
        outputs=[result_headline, tale_of_the_tape_panel, rematch_history],
    )

    @demo.load(outputs=[model_dropdown, division_dropdown])
    def on_load():
        models = _get("/models")
        divisions = _get("/divisions")
        return (
            gr.update(choices=[(m["label"], m["id"]) for m in models], value=models[0]["id"]),
            gr.update(choices=divisions, value=divisions[0]),
        )


if __name__ == "__main__":
    demo.launch()
