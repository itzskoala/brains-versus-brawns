from frontend import images as img

# Real markup shapes confirmed against live ufc.com fetches on 2026-10-14:
# a real athlete page carries <meta property="og:image" content="...">
# and a final URL containing /athlete/<slug>; a fighter with a profile but
# no uploaded photo has the same /athlete/ URL but no og:image tag at all;
# a fighter with no profile redirects (ufc.com's own redirect, followed by
# requests) to /search?... - never /athlete/.
REAL_ATHLETE_PAGE_HTML = """
<html><head>
<meta property="og:title" content="Sean O'Malley | UFC" />
<meta property="og:image" content="https://ufc.com/images/2026-06/OMALLEY_SEAN_06-14.png" />
</head><body></body></html>
"""
ATHLETE_PAGE_NO_PHOTO_HTML = """
<html><head>
<meta property="og:title" content="Sean Bassett | UFC" />
</head><body></body></html>
"""


def test_slugify_simple_name():
    assert img._slugify_name("Jon Jones") == "jon-jones"


def test_slugify_strips_apostrophes_not_hyphens():
    assert img._slugify_name("Sean O'Malley") == "sean-omalley"


def test_slugify_three_word_name():
    assert img._slugify_name("Ian Machado Garry") == "ian-machado-garry"


def test_parse_og_image_found():
    url = img._parse_og_image(REAL_ATHLETE_PAGE_HTML, "https://www.ufc.com/athlete/sean-omalley")
    assert url == "https://ufc.com/images/2026-06/OMALLEY_SEAN_06-14.png"


def test_parse_og_image_profile_exists_but_no_photo():
    # A real, 200-status /athlete/ page that simply has no og:image yet -
    # common for fighters predating UFC.com's current athlete database.
    url = img._parse_og_image(ATHLETE_PAGE_NO_PHOTO_HTML, "https://www.ufc.com/athlete/sean-bassett")
    assert url is None


def test_parse_og_image_redirected_to_search_is_not_found():
    # ufc.com's own redirect for a nonexistent slug - final URL has no
    # /athlete/ in it even though this HTML happens to carry an og:image
    # (a generic site-search one) - must not be mistaken for a real photo.
    html = REAL_ATHLETE_PAGE_HTML  # deliberately reuse: the URL check must dominate, not the HTML content
    url = img._parse_og_image(html, "https://www.ufc.com/search?query=athlete+nobody")
    assert url is None


def test_parse_og_image_malformed_html_does_not_crash():
    assert img._parse_og_image("<html><body>not even close</body></html>",
                                "https://www.ufc.com/athlete/whoever") is None


class _FakeResponse:
    def __init__(self, status_code, text="", url="", json_data=None):
        self.status_code = status_code
        self.text = text
        self.url = url
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("no JSON body")
        return self._json_data


class _FakeSession:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.requested_urls = []

    def get(self, url, headers=None, timeout=None):
        self.requested_urls.append(url)
        if self.raises:
            raise self.raises
        return self.response


def test_fetch_fighter_image_url_success():
    session = _FakeSession(_FakeResponse(200, REAL_ATHLETE_PAGE_HTML, "https://www.ufc.com/athlete/sean-omalley"))
    url = img.fetch_fighter_image_url("Sean O'Malley", session=session)
    assert url == "https://ufc.com/images/2026-06/OMALLEY_SEAN_06-14.png"
    assert session.requested_urls == ["https://www.ufc.com/athlete/sean-omalley"]


def test_fetch_fighter_image_url_http_error_status_returns_none():
    session = _FakeSession(_FakeResponse(500, "", "https://www.ufc.com/athlete/x"))
    assert img.fetch_fighter_image_url("Someone", session=session) is None


def test_fetch_fighter_image_url_network_failure_returns_none_not_raise():
    import requests
    session = _FakeSession(raises=requests.ConnectionError("no network"))
    assert img.fetch_fighter_image_url("Someone", session=session) is None


def test_fetch_fighter_image_url_timeout_returns_none_not_raise():
    import requests
    session = _FakeSession(raises=requests.Timeout("too slow"))
    assert img.fetch_fighter_image_url("Someone", session=session) is None


# Real Wikipedia REST API response shapes confirmed live on 2026-10-14
# (en.wikipedia.org/api/rest_v1/page/summary/<title>).
WIKIPEDIA_VOLKANOVSKI_JSON = {
    "type": "standard",
    "title": "Alexander Volkanovski",
    "description": "Australian mixed martial artist (born 1988)",
    "thumbnail": {"source": "https://thumb.wikimedia.org/.../Volkanovski.jpeg", "width": 330, "height": 382},
}
WIKIPEDIA_DISAMBIGUATION_JSON = {
    "type": "disambiguation",
    "title": "Michael Johnson",
    "description": None,
}
WIKIPEDIA_UNRELATED_PERSON_JSON = {
    "type": "standard",
    "title": "Someone Else",
    "description": "American actor (born 1970)",
    "thumbnail": {"source": "https://thumb.wikimedia.org/.../someone-else.jpeg"},
}
WIKIPEDIA_NO_THUMBNAIL_JSON = {
    "type": "standard",
    "title": "Obscure Fighter",
    "description": "American mixed martial artist",
}


def test_parse_wikipedia_summary_real_fighter():
    url = img._parse_wikipedia_summary(WIKIPEDIA_VOLKANOVSKI_JSON)
    assert url == "https://thumb.wikimedia.org/.../Volkanovski.jpeg"


def test_parse_wikipedia_summary_disambiguation_page_is_not_a_match():
    # A conflicted name (several notable people share it) - must not guess.
    assert img._parse_wikipedia_summary(WIKIPEDIA_DISAMBIGUATION_JSON) is None


def test_parse_wikipedia_summary_standard_article_but_not_a_fighter_is_not_a_match():
    # A real, unambiguous article - but about an unrelated same-named
    # person (no "mixed martial artist" in the description). A wrong photo
    # is worse than no photo.
    assert img._parse_wikipedia_summary(WIKIPEDIA_UNRELATED_PERSON_JSON) is None


def test_parse_wikipedia_summary_no_thumbnail_is_not_a_match():
    assert img._parse_wikipedia_summary(WIKIPEDIA_NO_THUMBNAIL_JSON) is None


def test_fetch_fighter_image_url_wikipedia_success():
    session = _FakeSession(_FakeResponse(200, json_data=WIKIPEDIA_VOLKANOVSKI_JSON))
    url = img.fetch_fighter_image_url_wikipedia("Alexander Volkanovski", session=session)
    assert url == "https://thumb.wikimedia.org/.../Volkanovski.jpeg"
    assert session.requested_urls == ["https://en.wikipedia.org/api/rest_v1/page/summary/Alexander_Volkanovski"]


def test_fetch_fighter_image_url_wikipedia_404_returns_none():
    session = _FakeSession(_FakeResponse(404, json_data={"status": 404}))
    assert img.fetch_fighter_image_url_wikipedia("Nobody At All", session=session) is None


def test_fetch_fighter_image_url_wikipedia_network_failure_returns_none_not_raise():
    import requests
    session = _FakeSession(raises=requests.Timeout("too slow"))
    assert img.fetch_fighter_image_url_wikipedia("Someone", session=session) is None


def test_fetch_fighter_image_wikipedia_title_encodes_spaces_as_underscores():
    session = _FakeSession(_FakeResponse(200, json_data=WIKIPEDIA_VOLKANOVSKI_JSON))
    img.fetch_fighter_image_url_wikipedia("Ian Machado Garry", session=session)
    assert session.requested_urls == ["https://en.wikipedia.org/api/rest_v1/page/summary/Ian_Machado_Garry"]


def test_fetch_fighter_image_tries_ufc_first_and_skips_wikipedia_on_success():
    ufc_session = _FakeSession(_FakeResponse(200, REAL_ATHLETE_PAGE_HTML, "https://www.ufc.com/athlete/sean-omalley"))
    wikipedia_session = _FakeSession(_FakeResponse(200, json_data=WIKIPEDIA_VOLKANOVSKI_JSON))
    url = img.fetch_fighter_image("Sean O'Malley", ufc_session=ufc_session, wikipedia_session=wikipedia_session)
    assert url == "https://ufc.com/images/2026-06/OMALLEY_SEAN_06-14.png"
    assert wikipedia_session.requested_urls == []  # never called - ufc.com already answered


def test_fetch_fighter_image_falls_back_to_wikipedia_when_ufc_has_no_photo():
    ufc_session = _FakeSession(_FakeResponse(200, ATHLETE_PAGE_NO_PHOTO_HTML, "https://www.ufc.com/athlete/royce-gracie"))
    wikipedia_session = _FakeSession(_FakeResponse(200, json_data=WIKIPEDIA_VOLKANOVSKI_JSON))
    url = img.fetch_fighter_image("Royce Gracie", ufc_session=ufc_session, wikipedia_session=wikipedia_session)
    assert url == "https://thumb.wikimedia.org/.../Volkanovski.jpeg"


def test_fetch_fighter_image_none_when_neither_source_has_a_confident_match():
    ufc_session = _FakeSession(_FakeResponse(200, ATHLETE_PAGE_NO_PHOTO_HTML, "https://www.ufc.com/athlete/x"))
    wikipedia_session = _FakeSession(_FakeResponse(200, json_data=WIKIPEDIA_DISAMBIGUATION_JSON))
    assert img.fetch_fighter_image("Nobody", ufc_session=ufc_session, wikipedia_session=wikipedia_session) is None


def test_get_fighter_image_url_caches_after_first_lookup():
    calls = []

    def fake_fetch(name):
        calls.append(name)
        return "https://ufc.com/images/x.png"

    cache = {}
    first = img.get_fighter_image_url("id1", "Jon Jones", cache=cache, fetch=fake_fetch)
    second = img.get_fighter_image_url("id1", "Jon Jones", cache=cache, fetch=fake_fetch)
    assert first == second == "https://ufc.com/images/x.png"
    assert calls == ["Jon Jones"]  # only fetched once


def test_get_fighter_image_url_caches_a_miss_too():
    # A fighter confirmed to have no photo must not be re-requested from
    # ufc.com on every subsequent selection in the same session.
    calls = []

    def fake_fetch(name):
        calls.append(name)
        return None

    cache = {}
    img.get_fighter_image_url("id2", "Obscure Fighter", cache=cache, fetch=fake_fetch)
    img.get_fighter_image_url("id2", "Obscure Fighter", cache=cache, fetch=fake_fetch)
    assert calls == ["Obscure Fighter"]
    assert cache["id2"] is None


def test_get_fighter_image_url_different_fighters_both_fetched():
    calls = []

    def fake_fetch(name):
        calls.append(name)
        return f"https://ufc.com/images/{name}.png"

    cache = {}
    img.get_fighter_image_url("id1", "Jon Jones", cache=cache, fetch=fake_fetch)
    img.get_fighter_image_url("id2", "Daniel Cormier", cache=cache, fetch=fake_fetch)
    assert calls == ["Jon Jones", "Daniel Cormier"]


def test_cache_persists_to_disk_and_reloads_across_restarts(tmp_path):
    # "Across all users, across restarts" - the whole point of persisting:
    # a brand new in-memory cache, loaded from a file another "run" wrote,
    # must already have the answer with zero fetch calls.
    cache_file = tmp_path / "fighter_image_cache.json"
    calls = []

    def fake_fetch(name):
        calls.append(name)
        return "https://ufc.com/images/jones.png"

    first_run_cache = img._load_cache_from_disk(cache_file)
    assert first_run_cache == {}
    img.get_fighter_image_url("id1", "Jon Jones", cache=first_run_cache, fetch=fake_fetch, persist=False)
    img._save_cache_to_disk(first_run_cache, cache_file)

    second_run_cache = img._load_cache_from_disk(cache_file)
    result = img.get_fighter_image_url("id1", "Jon Jones", cache=second_run_cache, fetch=fake_fetch, persist=False)
    assert result == "https://ufc.com/images/jones.png"
    assert calls == ["Jon Jones"]  # never fetched a second time, even in a fresh cache dict


def test_cache_persists_a_confirmed_miss_too():
    cache_file_cache = {}

    def fake_fetch(name):
        return None

    img.get_fighter_image_url("id2", "Obscure Fighter", cache=cache_file_cache, fetch=fake_fetch, persist=False)
    assert cache_file_cache == {"id2": None}


def test_load_cache_from_disk_missing_file_returns_empty_dict(tmp_path):
    assert img._load_cache_from_disk(tmp_path / "does-not-exist.json") == {}


def test_load_cache_from_disk_corrupt_file_returns_empty_dict_not_crash(tmp_path):
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{not valid json")
    assert img._load_cache_from_disk(bad_file) == {}


def test_save_cache_to_disk_write_failure_does_not_raise(tmp_path):
    # A read-only filesystem (some deploy targets) must degrade to
    # "re-fetch next time", never crash the app that's just trying to show
    # a fighter photo. Force a real failure: "blocker" is a FILE, so
    # mkdir(parents=True) under it cannot succeed.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    img._save_cache_to_disk({"id1": "x"}, blocker / "subdir" / "file.json")  # must not raise


def test_save_then_load_round_trips_exactly(tmp_path):
    cache_file = tmp_path / "cache.json"
    original = {"id1": "https://ufc.com/images/x.png", "id2": None}
    img._save_cache_to_disk(original, cache_file)
    assert img._load_cache_from_disk(cache_file) == original


def test_avatar_initials_two_word_name():
    assert img._avatar_initials("Sean O'Malley") == "SO"


def test_avatar_initials_blank_name_is_question_mark():
    assert img._avatar_initials("") == "?"
    assert img._avatar_initials(None) == "?"


def test_avatar_color_deterministic_for_same_name():
    assert img._avatar_color("Jon Jones") == img._avatar_color("Jon Jones")


def test_avatar_color_differs_for_different_names_usually():
    # Not a strict guarantee (small fixed palette, hashes can collide), but
    # these two should land on different colors - catches an accidental
    # "always return the first color" bug.
    assert img._avatar_color("Jon Jones") != img._avatar_color("Sean O'Malley")


def test_fighter_placeholder_html_has_no_broken_image_tag():
    html = img.fighter_placeholder_html()
    assert "<img" not in html
    assert "Select a fighter" in html


def test_fighter_loading_html_shows_fight_title_formatted_name():
    html = img.fighter_loading_html("Jon Jones", "Bones")
    assert "Loading photo" in html
    assert 'Jon &quot;Bones&quot; Jones' in html


def test_fighter_photo_html_with_real_image_renders_img_tag():
    html = img.fighter_photo_html("Sean O'Malley", "Suga", "https://ufc.com/images/x.png")
    assert '<img src="https://ufc.com/images/x.png"' in html
    assert "Suga" in html


def test_fighter_photo_html_missing_image_renders_initials_avatar_not_broken_img():
    html = img.fighter_photo_html("Sean Bassett", None, None)
    assert "<img" not in html
    assert "SB" in html
    assert "Sean Bassett" in html


def test_fighter_photo_html_with_real_image_has_no_onerror_callback_markup():
    # Deliberately simple: image_url was already confirmed live at lookup
    # time (see fetch_fighter_image_url), so no client-side onerror/hidden-
    # sibling-div fallback juggling - just a plain <img>.
    html = img.fighter_photo_html("Sean O'Malley", "Suga", "https://ufc.com/images/x.png")
    assert "onerror" not in html
    assert html.count("<img") == 1
    assert html.count("<div") == 2  # outer wrapper + the name caption, nothing else
