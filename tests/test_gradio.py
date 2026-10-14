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
