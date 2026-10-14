import pytest

from src.data.scraping.ufcstats_parse import (
    ChallengePageError,
    ParseError,
    parse_completed_events_list,
    parse_event_page,
    parse_fight_page,
    parse_fighter_page,
)

# All markup below is trimmed to the exact classes/structure confirmed
# against real UFCStats.com pages fetched via headless Chromium on
# 2026-10-14 (completed-events list, UFC 330: Makhachev vs. Machado Garry's
# event page, one of its fight-details pages, and one fighter-details
# page) - not guessed selectors.

CHALLENGE_HTML = """
<!doctype html><html><head><title>Loading…</title>
<meta name="robots" content="noindex"></head>
<body><p>Checking your browser…</p><noscript>This site requires JavaScript.</noscript></body></html>
"""

COMPLETED_EVENTS_HTML = """
<table class="b-statistics__table-events">
  <tr class="b-statistics__table-row"><th class="b-statistics__table-col">Name/date</th></tr>
  <tr class="b-statistics__table-row">
    <td class="b-statistics__table-col">
      <i class="b-statistics__table-content">
        <a class="b-link b-link_style_black" href="http://ufcstats.com/event-details/7f98d9d5a10fa25c">
          UFC Fight Night: Allen vs. Duncan</a>
        <span class="b-statistics__date">October 10, 2026</span>
      </i>
    </td>
    <td class="b-statistics__table-col">Las Vegas, Nevada, USA</td>
  </tr>
  <tr class="b-statistics__table-row">
    <td class="b-statistics__table-col">
      <i class="b-statistics__table-content">
        <a class="b-link b-link_style_black" href="http://ufcstats.com/event-details/ad3fdba28a7540cf">
          UFC 332: Silva vs. Wang</a>
        <span class="b-statistics__date">October 03, 2026</span>
      </i>
    </td>
    <td class="b-statistics__table-col">Las Vegas, Nevada, USA</td>
  </tr>
</table>
"""


def test_parse_completed_events_list():
    events = parse_completed_events_list(COMPLETED_EVENTS_HTML, "test")
    assert [e["event_id"] for e in events] == ["7f98d9d5a10fa25c", "ad3fdba28a7540cf"]
    assert events[0]["event_name"] == "UFC Fight Night: Allen vs. Duncan"
    assert events[0]["event_date"].isoformat() == "2026-10-10"
    assert events[0]["location"] == "Las Vegas, Nevada, USA"


def test_completed_events_list_challenge_page_raises_challenge_error():
    with pytest.raises(ChallengePageError):
        parse_completed_events_list(CHALLENGE_HTML, "test")


def test_completed_events_list_malformed_html_raises_parse_error():
    with pytest.raises(ParseError):
        parse_completed_events_list("<html><body>not the right page</body></html>", "test")


EVENT_PAGE_HTML = """
<h2 class="b-content__title"><span class="b-content__title-highlight">
  UFC 330: Makhachev vs. Machado Garry</span></h2>
<ul class="b-list__box-list">
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Date:</i> August 15, 2026</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Location:</i> Philadelphia, Pennsylvania, USA</li>
</ul>
<table class="b-fight-details__table_type_event-details">
  <tr data-link="http://ufcstats.com/fight-details/365fd759c03e93cf">
    <td><p>win</p></td>
    <td><a href="http://ufcstats.com/fighter-details/275aca31f61ba28c">Islam Makhachev</a>
        <a href="http://ufcstats.com/fighter-details/442c9011034ae1fd">Ian Machado Garry</a></td>
    <td>1<br>0</td><td>22<br>29</td><td>7<br>0</td><td>0<br>0</td>
    <td><p>Welterweight<br><img src="http://cdn/belt.png"></p></td>
    <td>U-DEC</td><td>5</td><td>5:00</td>
  </tr>
  <tr data-link="http://ufcstats.com/fight-details/d4bd456d70965734">
    <td><p>win</p></td>
    <td><a href="http://ufcstats.com/fighter-details/4d8fa64202cee4c1">Jalin Turner</a>
        <a href="http://ufcstats.com/fighter-details/ec13c393d029297d">Kaue Fernandes</a></td>
    <td>0<br>0</td><td>10<br>5</td><td>0<br>0</td><td>0<br>0</td>
    <td><p>Lightweight<br><img src="http://cdn/perf.png"></p></td>
    <td>SUB</td><td>2</td><td>3:12</td>
  </tr>
</table>
"""


def test_parse_event_page_basics():
    ev = parse_event_page(EVENT_PAGE_HTML, "test")
    assert ev["event_name"] == "UFC 330: Makhachev vs. Machado Garry"
    assert ev["event_date"].isoformat() == "2026-08-15"
    assert ev["location"] == "Philadelphia, Pennsylvania, USA"
    assert len(ev["fights"]) == 2


def test_parse_event_page_title_fight_from_belt_icon():
    ev = parse_event_page(EVENT_PAGE_HTML, "test")
    title_fight = next(f for f in ev["fights"] if f["fight_id"] == "365fd759c03e93cf")
    non_title = next(f for f in ev["fights"] if f["fight_id"] == "d4bd456d70965734")
    assert title_fight["title_fight"] is True
    assert non_title["title_fight"] is False


def test_parse_event_page_bonus_from_perf_icon():
    ev = parse_event_page(EVENT_PAGE_HTML, "test")
    bonus_fight = next(f for f in ev["fights"] if f["fight_id"] == "d4bd456d70965734")
    assert bonus_fight["bonus_types"] == ["Performance of the Night"]


def test_parse_event_page_does_not_return_corner_assignment():
    # Deliberate: the event page's fighter order was found to disagree with
    # fight-details pages (and the existing dataset) often enough that it's
    # not trusted for r_id/b_id - see ufcstats_parse.parse_event_page's
    # docstring. Fight dicts here must not expose fighter identity at all.
    ev = parse_event_page(EVENT_PAGE_HTML, "test")
    assert "r_fighter_id" not in ev["fights"][0]
    assert "b_fighter_id" not in ev["fights"][0]


def test_parse_event_page_challenge_page_raises():
    with pytest.raises(ChallengePageError):
        parse_event_page(CHALLENGE_HTML, "test")


def _fight_page(include_round_tables):
    totals_table = """
    <table><tr><th>Fighter</th><th>KD</th><th>Sig. str.</th><th>Sig. str. %</th>
      <th>Total str.</th><th>Td %</th><th>Td %</th><th>Sub. att</th><th>Rev.</th><th>Ctrl</th></tr>
      <tbody class="b-fight-details__table-body"><tr>
        <td><p>Rafael Tobias</p><p>Lucas Fernando</p></td>
        <td><p>0</p><p>2</p></td>
        <td><p>13 of 56</p><p>69 of 115</p></td>
        <td><p>23%</p><p>60%</p></td>
        <td><p>14 of 58</p><p>69 of 115</p></td>
        <td><p>1 of 1</p><p>0 of 0</p></td>
        <td><p>100%</p><p>---</p></td>
        <td><p>0</p><p>0</p></td>
        <td><p>0</p><p>0</p></td>
        <td><p>0:55</p><p>0:09</p></td>
      </tr></tbody>
    </table>
    """
    target_table = """
    <table><tr><th>Fighter</th><th>Sig. str</th><th>Sig. str. %</th><th>Head</th><th>Body</th>
      <th>Leg</th><th>Distance</th><th>Clinch</th><th>Ground</th></tr>
      <tbody class="b-fight-details__table-body"><tr>
        <td><p>Rafael Tobias</p><p>Lucas Fernando</p></td>
        <td><p>13 of 56</p><p>69 of 115</p></td>
        <td><p>23%</p><p>60%</p></td>
        <td><p>5 of 47</p><p>37 of 80</p></td>
        <td><p>5 of 5</p><p>18 of 20</p></td>
        <td><p>3 of 4</p><p>14 of 15</p></td>
        <td><p>11 of 52</p><p>66 of 111</p></td>
        <td><p>1 of 2</p><p>1 of 1</p></td>
        <td><p>1 of 2</p><p>2 of 3</p></td>
      </tr></tbody>
    </table>
    """
    per_round_totals = per_round_target = ""
    if include_round_tables:
        per_round_totals = totals_table.replace(
            '<tbody class="b-fight-details__table-body"><tr>',
            '<tbody class="b-fight-details__table-body">'
            '<tr><th>Round 1</th></tr><tr>'
        )
        per_round_target = target_table.replace(
            '<tbody class="b-fight-details__table-body"><tr>',
            '<tbody class="b-fight-details__table-body">'
            '<tr><th>Round 1</th></tr><tr>'
        )

    # Real UFCStats page order (verified 2026-10-14): totals-overall,
    # totals-per-round, sig-by-target-overall, sig-by-target-per-round -
    # the two per-round tables are NOT grouped together at the end.
    return f"""
    <div class="b-fight-details__persons clearfix">
      <div class="b-fight-details__person">
        <i class="b-fight-details__person-status">L</i>
        <a class="b-fight-details__person-link" href="http://ufcstats.com/fighter-details/d6a8edb7410223a0">Rafael Tobias</a>
      </div>
      <div class="b-fight-details__person">
        <i class="b-fight-details__person-status">W</i>
        <a class="b-fight-details__person-link" href="http://ufcstats.com/fighter-details/3df223aeaacecf3f">Lucas Fernando</a>
      </div>
    </div>
    <p>
      <i class="b-fight-details__text-item_first"><i class="b-fight-details__label">Method:</i> KO/TKO</i>
      <i class="b-fight-details__text-item"><i class="b-fight-details__label">Round:</i> 3</i>
      <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time:</i> 1:10</i>
      <i class="b-fight-details__text-item"><i class="b-fight-details__label">Time format:</i> 3 Rnd (5-5-5)</i>
      <i class="b-fight-details__text-item"><i class="b-fight-details__label">Referee:</i> <span>Keith Peterson</span></i>
    </p>
    <p><i class="b-fight-details__label">Details:</i> Knee to Body At Distance</p>
    {totals_table}
    {per_round_totals}
    {target_table}
    {per_round_target}
    """


def test_parse_fight_page_winner_and_corners():
    fight = parse_fight_page(_fight_page(include_round_tables=False), "f1", "test")
    assert fight["r_fighter_id"] == "d6a8edb7410223a0"
    assert fight["b_fighter_id"] == "3df223aeaacecf3f"
    assert fight["winner_id"] == "3df223aeaacecf3f"
    assert fight["result_status"] == "win"
    assert fight["method"] == "KO/TKO"
    assert fight["round"] == 3


def test_parse_fight_page_totals():
    fight = parse_fight_page(_fight_page(include_round_tables=False), "f1", "test")
    assert fight["totals"]["r_total_sig_landed"] == 13
    assert fight["totals"]["r_total_sig_atmp"] == 56
    assert fight["totals"]["b_total_td_success"] == 0
    assert fight["totals"]["r_total_ctrl_seconds"] == 55
    assert fight["totals"]["r_total_sig_str_landed_head"] == 5


def test_parse_fight_page_no_round_tables_is_tolerated_not_an_error():
    # Historically documented gap (data/unused/scrape_error.csv:
    # "expected 2 round-by-round tables, found 0") - a fight with only the
    # two fight-level tables must parse fine, just with no round rows.
    fight = parse_fight_page(_fight_page(include_round_tables=False), "f1", "test")
    assert fight["round_rows"] == []


def test_parse_fight_page_with_round_tables():
    fight = parse_fight_page(_fight_page(include_round_tables=True), "f1", "test")
    assert len(fight["round_rows"]) == 1
    rr = fight["round_rows"][0]
    assert rr["round_no"] == 1
    assert rr["r_sig_landed"] == 13
    assert rr["r_ctrl"] == "0:55"


def test_parse_fight_page_challenge_page_raises():
    with pytest.raises(ChallengePageError):
        parse_fight_page(CHALLENGE_HTML, "f1", "test")


def test_parse_fight_page_missing_persons_raises_parse_error():
    html = "<html><body><table></table><table></table></body></html>"
    with pytest.raises(ParseError):
        parse_fight_page(html, "f1", "test")


def test_parse_fight_page_malformed_stat_table_raises():
    # Persons/head present, but only one stat table instead of 2 or 4.
    broken = _fight_page(include_round_tables=False).rsplit("<table>", 1)[0]
    with pytest.raises(ParseError):
        parse_fight_page(broken, "f1", "test")


FIGHTER_PAGE_HTML = """
<h2 class="b-content__title"><span class="b-content__title-highlight">Tresean Gore</span>
<span class="b-content__title-record">Record: 8-4-0</span></h2>
<p class="b-content__Nickname">Mr. Vicious</p>
<ul class="b-list__box-list">
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Height:</i> 6' 0"</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Weight:</i> 185 lbs.</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Reach:</i> 75"</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">STANCE:</i> Switch</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">DOB:</i> Jun 21, 1994</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">SLpM:</i> 3.59</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Str. Acc.:</i> 49%</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">SApM:</i> 6.24</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Str. Def:</i> 43%</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Avg.:</i> 1.54</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Acc.:</i> 61%</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">TD Def.:</i> 84%</li>
  <li class="b-list__box-list-item"><i class="b-list__box-item-title">Sub. Avg.:</i> 1.0</li>
</ul>
"""


def test_parse_fighter_page():
    fighter = parse_fighter_page(FIGHTER_PAGE_HTML, "147e70aa6cc48cfc", "test")
    assert fighter["fighter_name"] == "Tresean Gore"
    assert fighter["fighter_nick_name"] == "Mr. Vicious"
    assert fighter["height"] == "6' 0\""
    assert fighter["weight_lbs"] == 185
    assert fighter["reach_inches"] == 75
    assert fighter["stance"] == "Switch"
    assert fighter["dob"] == "1994-06-21"
    assert fighter["str_acc"] == 49
    assert fighter["slpm"] == 3.59


def test_parse_fighter_page_blank_fields_become_none_not_crash():
    html = FIGHTER_PAGE_HTML.replace('Switch', '--').replace('75"', '--')
    fighter = parse_fighter_page(html, "x", "test")
    assert fighter["stance"] is None
    assert fighter["reach_inches"] is None


def test_parse_fighter_page_challenge_page_raises():
    with pytest.raises(ChallengePageError):
        parse_fighter_page(CHALLENGE_HTML, "x", "test")
