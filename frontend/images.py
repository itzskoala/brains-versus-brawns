"""Fighter photos for the matchup screen. Free, no paid API, two sources
tried in order:

1. ufc.com has a public athlete-profile page per fighter at a predictable
   /athlete/<first-last> URL, and (confirmed live, 2026-10-14) ufc.com's
   own robots.txt allows /athlete, has no bot-wall (unlike UFCStats - see
   src/data/scraping/ufcstats_browser.py), and redirects a close-but-wrong
   slug to the correct one or, for a fighter with no profile at all, to
   /search - a clean, cheap "do we have a photo for this person" signal
   with no guessing required. In production (deployed on shared/datacenter
   IPs rather than a home connection) ufc.com can be flakier than it was
   in local testing - likely stricter treatment of cloud IP ranges, same
   genus of issue as the rate-limiting independently confirmed during
   development (a plain curl that was ~1s earlier took 64s+ after heavy
   testing volume). Any failure here - timeout, non-200, no photo - falls
   through to:
2. Wikipedia's REST summary API (en.wikipedia.org/api/rest_v1/page/summary/
   <title>) - explicitly built for and tolerant of automated/bot traffic
   (see their API etiquette policy; robots.txt has no AI-crawler
   restriction, unlike e.g. espn.com's explicit site-wide "anthropic-ai:
   Disallow /"), so it works reliably even where ufc.com doesn't. A plain
   name lookup can land on an unrelated same-named person (there could be
   a "Michael Johnson" who isn't our fighter) - handled conservatively:
   only trust the result when Wikipedia calls it a "standard" (not
   disambiguation) article AND its own one-line description contains
   "mixed martial artist". A wrong photo is worse than no photo, so this
   skips rather than guesses when unsure.

Every CONFIRMED, LEGITIMATE image URL - and only that, never a miss and
never a fallback avatar - is cached by fighter_id on disk at
CACHE_FILE_PATH, not just in memory - loaded once at import and appended
to on every new successful lookup, so a fighter with a real photo is only
ever looked up once, ever, across every future run and every user of this
app, not just once per process. A miss (no photo found, or any fetch
failure - a timeout, a non-200, rate-limiting) is deliberately never
cached, so it's simply retried on the next lookup rather than permanently
locking a fighter out of ever getting their real photo because of one bad
network moment. See get_fighter_image_url.

Most fighters in this dataset (anyone who fought before UFC.com's current
athlete-database era and has no Wikipedia page either) won't have a photo
at all - that's expected, not an error, and handled by falling back to a
generated initials avatar (fighter_photo_html), never a broken image or a
crash.
"""
import hashlib
import html as html_lib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

ATHLETE_URL = "https://www.ufc.com/athlete/{slug}"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
# Wikipedia's API etiquette explicitly asks for an identifying User-Agent
# (the opposite of ufc.com's USER_AGENT above, which blends in as a normal
# browser) - https://www.mediawiki.org/wiki/API:Etiquette.
WIKIPEDIA_USER_AGENT = "ufc-elo-matchup-predictor/1.0 (personal ML project; contact: sri.kotala@gmail.com)"
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
    is a nice-to-have, never worth crashing the matchup screen over).
    Logs the outcome either way (status/timing/exception) - this is the
    only observability into production fetch behavior there is, since
    HF Space logs otherwise only show the startup line."""
    session = session or requests
    slug = _slugify_name(fighter_name)
    url = ATHLETE_URL.format(slug=slug)
    t0 = time.monotonic()
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    except requests.RequestException as exc:
        print(f"[images] ufc.com FAILED fighter={fighter_name!r} url={url} "
              f"elapsed={time.monotonic() - t0:.2f}s error={type(exc).__name__}: {exc}")
        return None
    elapsed = time.monotonic() - t0
    if resp.status_code != 200:
        print(f"[images] ufc.com non-200 fighter={fighter_name!r} url={url} "
              f"status={resp.status_code} elapsed={elapsed:.2f}s")
        return None
    image_url = _parse_og_image(resp.text, resp.url)
    print(f"[images] ufc.com OK fighter={fighter_name!r} url={url} final_url={resp.url} "
          f"status=200 elapsed={elapsed:.2f}s image={'found' if image_url else 'none'}")
    return image_url


def _parse_wikipedia_summary(data):
    """Wikipedia summary JSON -> image URL, or None if this isn't
    confidently the right person. A plain-name lookup can land on an
    unrelated same-named person (there could be a non-fighter "Michael
    Johnson"), so this requires BOTH: a real standalone article (type ==
    "standard", not "disambiguation" - which is what a conflicted name
    actually returns, confirmed live for "Michael Johnson") AND the
    article's own one-line description containing "mixed martial artist"
    (confirmed present on every real UFC fighter's page checked, e.g.
    "Australian mixed martial artist (born 1988)" for Volkanovski). A
    wrong photo is worse than no photo - this skips rather than guesses."""
    if data.get("type") != "standard":
        return None
    description = (data.get("description") or "").lower()
    if "mixed martial artist" not in description:
        return None
    url = (data.get("thumbnail") or {}).get("source")
    return url if url and url.startswith("http") else None


def fetch_fighter_image_url_wikipedia(fighter_name, session=None):
    """One live Wikipedia lookup - no cache, no retry. Returns the photo
    URL or None (no matching article, not confidently this person, or any
    network failure). Logged the same way as fetch_fighter_image_url."""
    session = session or requests
    title = quote(fighter_name.strip().replace(" ", "_"))
    url = WIKIPEDIA_SUMMARY_URL.format(title=title)
    t0 = time.monotonic()
    try:
        resp = session.get(url, headers={"User-Agent": WIKIPEDIA_USER_AGENT}, timeout=15)
    except requests.RequestException as exc:
        print(f"[images] wikipedia FAILED fighter={fighter_name!r} url={url} "
              f"elapsed={time.monotonic() - t0:.2f}s error={type(exc).__name__}: {exc}")
        return None
    elapsed = time.monotonic() - t0
    if resp.status_code != 200:
        print(f"[images] wikipedia non-200 fighter={fighter_name!r} url={url} "
              f"status={resp.status_code} elapsed={elapsed:.2f}s")
        return None
    try:
        data = resp.json()
    except ValueError as exc:
        print(f"[images] wikipedia bad JSON fighter={fighter_name!r} url={url} error={exc}")
        return None
    image_url = _parse_wikipedia_summary(data)
    print(f"[images] wikipedia OK fighter={fighter_name!r} url={url} status=200 "
          f"elapsed={elapsed:.2f}s type={data.get('type')} image={'found' if image_url else 'none'}")
    return image_url


def fetch_fighter_image(fighter_name, ufc_session=None, wikipedia_session=None):
    """The default two-source lookup: ufc.com first (the UFC's own photo),
    then Wikipedia as a free, bot-tolerant second source for fighters
    UFC.com has none for (or where UFC.com itself is unreachable - see this
    module's docstring on production/datacenter-IP flakiness). Returns the
    first confident match, or None if neither has one."""
    return (fetch_fighter_image_url(fighter_name, session=ufc_session)
            or fetch_fighter_image_url_wikipedia(fighter_name, session=wikipedia_session))


def get_fighter_image_url(fighter_id, fighter_name, cache=None, fetch=fetch_fighter_image, persist=True):
    """Cached wrapper around fetch_fighter_image (ufc.com, then Wikipedia) -
    the only entry point the UI should call. `cache` defaults to this
    module's disk-backed, process-wide dict (shared by every user of this
    running app, not per-session); tests inject their own in-memory dict
    and pass persist=False.

    Only a confirmed, legitimate image URL is ever cached - a miss (no
    photo from either source, a timeout, a non-200, rate-limiting, any
    failure at all) is NEVER written to the cache, deliberately. Caching a
    miss would mean one bad network moment (confirmed to happen in
    production - see frontend/images.py's module docstring) permanently
    stamps a real fighter as "no photo", with no way to self-correct short
    of someone manually editing the cache file. Leaving misses uncached
    means they're simply retried on the next lookup - a small repeated
    cost for fighters who truly have no photo anywhere, in exchange for
    never getting permanently and silently stuck wrong for one who does."""
    cache = _image_url_cache if cache is None else cache
    if fighter_id in cache:
        return cache[fighter_id]
    image_url = fetch(fighter_name)
    if image_url:
        cache[fighter_id] = image_url
        if persist and cache is _image_url_cache:
            _save_cache_to_disk(cache)
    return image_url


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
    """The resolved state: a real photo (ufc.com or Wikipedia) if we found
    one, otherwise a deterministic initials avatar - never blank. image_url
    was already confirmed live at lookup time (see fetch_fighter_image), so
    this just renders it directly - no client-side onerror fallback
    juggling a hidden sibling div."""
    name = html_lib.escape(format_fighter_name(fighter_name, nickname))
    if image_url:
        safe_url = html_lib.escape(image_url)
        box = _photo_box_style("object-fit:contain;background:#1a1a1a;")
        img = f'<img src="{safe_url}" alt="{name}" style="{box}">'
    else:
        box = _photo_box_style(f"background:{_avatar_color(fighter_name)};color:white;font-size:48px;font-weight:700;")
        img = f'<div style="{box}">{html_lib.escape(_avatar_initials(fighter_name))}</div>'
    return f'<div style="text-align:center;">{img}<div style="margin-top:8px;font-weight:600;">{name}</div></div>'
