"""Fighter photos for the matchup screen. Free, no paid API: ufc.com has a
public athlete-profile page per fighter at a predictable
/athlete/<first-last> URL, and (confirmed live, 2026-10-14) ufc.com's own
robots.txt allows /athlete, has no bot-wall (unlike UFCStats - see
src/data/scraping/ufcstats_browser.py), and redirects a close-but-wrong
slug to the correct one or, for a fighter with no profile at all, to
/search - a clean, cheap "do we have a photo for this person" signal with
no guessing required.

Every resolved URL (including a confirmed "no photo") is cached by
fighter_id on disk at CACHE_FILE_PATH, not just in memory - loaded once at
import and appended to on every new lookup, so a fighter is only ever
looked up on ufc.com ONCE, ever, across every future run and every user of
this app, not just once per process. See get_fighter_image_url.

Most fighters in this dataset (anyone who fought before UFC.com's current
athlete-database era, or who simply never got a profile page) won't have a
photo here at all - that's expected, not an error, and handled by falling
back to a generated initials avatar (fighter_photo_html), never a broken
image or a crash.
"""
import hashlib
import html as html_lib
import json
import re
from pathlib import Path

import requests

ATHLETE_URL = "https://www.ufc.com/athlete/{slug}"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
PHOTO_SIZE_PX = 220
CACHE_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "fighter_image_cache.json"


def _load_cache_from_disk(path=CACHE_FILE_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache_to_disk(cache, path=CACHE_FILE_PATH):
    # A fighter photo is a nice-to-have, not critical data - a read-only
    # filesystem (some deploy targets) or any other write failure should
    # degrade to "re-fetch next time", never crash the app.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except OSError:
        pass


_image_url_cache = _load_cache_from_disk()


def split_name(full_name):
    """"Sean O'Malley" -> ("Sean", "O'Malley"); "Ian Machado Garry" ->
    ("Ian", "Machado Garry"); a single-token name -> (name, ""). The
    canonical home for this (and format_fighter_name below) is here, not
    frontend/gradio.py, specifically so gradio.py can depend on this module
    without this module ever depending back on gradio.py - a real circular
    import between the two broke the Blocks graph (ufc_elo issue, 2026-10:
    `python -m frontend.gradio` runs gradio.py as __main__, so a reverse
    import of it by its real package name re-executed its whole
    module-level code a second time mid-request)."""
    parts = (full_name or "").split(None, 1)
    if len(parts) < 2:
        return (parts[0] if parts else ""), ""
    return parts[0], parts[1]


def format_fighter_name(full_name, nickname):
    """"Sean O'Malley" + "Suga" -> 'Sean "Suga" O\\'Malley'. No nickname
    (None, "", or whitespace-only) -> plain "First Last" - never a stray
    pair of quotes for a fighter with no nickname on record."""
    first, last = split_name(full_name)
    nickname = (nickname or "").strip()
    if nickname:
        return f'{first} "{nickname}" {last}'.strip()
    return f"{first} {last}".strip()


def _slugify_name(full_name):
    """"Sean O'Malley" -> "sean-omalley"; "Ian Machado Garry" ->
    "ian-machado-garry". Doesn't need to be exactly right - ufc.com's own
    redirect handles a close guess (e.g. a skipped middle name) by sending
    us to the canonical slug; only a genuinely nonexistent fighter lands on
    /search, which _parse_og_image's caller treats as "no photo"."""
    slug = full_name.lower().replace("'", "").replace(".", "")
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def _parse_og_image(html_text, final_url):
    """ufc.com athlete page HTML -> its og:image URL, or None if this isn't
    really an athlete page (redirected to /search - no profile exists) or
    the page has no og:image (a profile exists but no photo was ever
    uploaded for them - common for fighters from before UFC.com's current
    athlete database)."""
    if "/athlete/" not in final_url:
        return None
    match = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html_text)
    if not match:
        return None
    url = html_lib.unescape(match.group(1))
    return url if url.startswith("http") else None


def fetch_fighter_image_url(fighter_name, session=None):
    """One live ufc.com lookup - no cache, no retry. Returns the photo URL
    or None (not found, no photo, or any network failure - a fighter photo
    is a nice-to-have, never worth crashing the matchup screen over)."""
    session = session or requests
    slug = _slugify_name(fighter_name)
    try:
        resp = session.get(ATHLETE_URL.format(slug=slug), headers={"User-Agent": USER_AGENT}, timeout=15)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return _parse_og_image(resp.text, resp.url)


def get_fighter_image_url(fighter_id, fighter_name, cache=None, fetch=fetch_fighter_image_url, persist=True):
    """Cached wrapper around fetch_fighter_image_url - the only entry point
    the UI should call. `cache` defaults to this module's disk-backed,
    process-wide dict (shared by every user of this running app, not
    per-session); tests inject their own in-memory dict and pass
    persist=False. A fighter with no photo is cached as None too - we never
    re-request ufc.com for them again, this run or any future one."""
    cache = _image_url_cache if cache is None else cache
    if fighter_id not in cache:
        cache[fighter_id] = fetch(fighter_name)
        if persist and cache is _image_url_cache:
            _save_cache_to_disk(cache)
    return cache[fighter_id]


_AVATAR_COLORS = ["#8B2E2E", "#2E5C8B", "#2E8B57", "#8B6F2E", "#5C2E8B", "#2E8B8B", "#8B2E6F"]


def _avatar_initials(full_name):
    """"Sean O'Malley" -> "SO". An empty/blank name -> "?" (never a crash,
    never an empty avatar)."""
    first, last = split_name(full_name)
    letters = (first[:1] + last[:1]).upper().strip()
    return letters or "?"


def _avatar_color(full_name):
    """Deterministic (same fighter -> same color every render, not a new
    random color each time) pick from a small fixed palette."""
    digest = hashlib.md5((full_name or "").encode()).hexdigest()
    return _AVATAR_COLORS[int(digest, 16) % len(_AVATAR_COLORS)]


def _photo_box_style(extra=""):
    return (f"width:{PHOTO_SIZE_PX}px;height:{PHOTO_SIZE_PX}px;border-radius:12px;"
            f"display:flex;align-items:center;justify-content:center;margin:0 auto;{extra}")


def fighter_placeholder_html():
    """Shown in a corner before any fighter is picked."""
    box = _photo_box_style("background:#2a2a2a;color:#666;font-size:14px;border:2px dashed #444;")
    return f'<div style="text-align:center;"><div style="{box}">Select a fighter</div></div>'


def fighter_loading_html(fighter_name, nickname):
    """Shown the instant a fighter is picked, before their photo lookup
    (and the rest of their stats) has come back - see predict's loading
    state in frontend/gradio.py."""
    box = _photo_box_style("background:#2a2a2a;color:#999;font-size:14px;")
    name = html_lib.escape(format_fighter_name(fighter_name, nickname))
    return (f'<div style="text-align:center;"><div style="{box}">Loading photo…</div>'
            f'<div style="margin-top:8px;font-weight:600;">{name}</div></div>')


def fighter_photo_html(fighter_name, nickname, image_url):
    """The resolved state: a real ufc.com photo if we found one, otherwise
    a deterministic initials avatar - never blank. image_url was already
    confirmed to come from a real 200-status ufc.com page at lookup time
    (see fetch_fighter_image_url), so this just renders it directly - no
    client-side onerror fallback juggling a hidden sibling div."""
    name = html_lib.escape(format_fighter_name(fighter_name, nickname))
    if image_url:
        safe_url = html_lib.escape(image_url)
        box = _photo_box_style("object-fit:contain;background:#1a1a1a;")
        img = f'<img src="{safe_url}" alt="{name}" style="{box}">'
    else:
        box = _photo_box_style(f"background:{_avatar_color(fighter_name)};color:white;font-size:48px;font-weight:700;")
        img = f'<div style="{box}">{html_lib.escape(_avatar_initials(fighter_name))}</div>'
    return f'<div style="text-align:center;">{img}<div style="margin-top:8px;font-weight:600;">{name}</div></div>'
