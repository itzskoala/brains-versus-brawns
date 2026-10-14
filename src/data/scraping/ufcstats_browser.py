"""UFCStats.com fetch layer. As of 2026-10, UFCStats serves a site-wide
client-side JS "Checking your browser..." gate on every page (confirmed via
a plain requests.get - homepage, a fighter page, and the completed-events
list all returned the same ~3KB challenge shell, not real content). A
default headless Chromium page - no stealth plugins, no fingerprint
spoofing, no proxy - clears it on its own within normal page-load
('networkidle'), the same way any browser executing the page's own JS
would: a single manual test against the completed-events list came back
with the real 30KB events table. That's the only thing this module does:
render the page like a browser and hand back the HTML. All parsing lives in
ufcstats_parse.py, which only ever sees plain HTML strings.

Kept deliberately polite: one browser/context for the whole pipeline run
(not relaunched per page), a real request fingerprint, and a short delay
between navigations - this runs a couple of times a week, not continuously.
"""
import time

from playwright.sync_api import sync_playwright

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
DELAY_SECONDS = 1.5


class UfcStatsFetchError(Exception):
    """Navigation to a ufcstats.com page failed outright (timeout, DNS, etc.)."""


class UfcStatsBrowser:
    """Context manager wrapping one headless-Chromium session.

        with UfcStatsBrowser() as browser:
            html = browser.get("http://ufcstats.com/statistics/events/completed")
    """

    def __init__(self, delay_seconds=DELAY_SECONDS):
        self.delay_seconds = delay_seconds
        self._playwright = None
        self._browser = None
        self._page = None
        self._first_request = True

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(user_agent=USER_AGENT)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    def get(self, url, wait_ms=1500):
        """Navigate to url and return the fully-rendered HTML. Raises
        UfcStatsFetchError on a hard navigation failure (not on a
        challenge/parse problem - ufcstats_parse.py's ParseError/
        ChallengePageError cover "we got a page back but it wasn't what we
        expected")."""
        if not self._first_request:
            time.sleep(self.delay_seconds)
        self._first_request = False

        try:
            response = self._page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception as exc:  # noqa: BLE001 - any Playwright navigation error
            raise UfcStatsFetchError(f"navigation to {url} failed: {exc}") from exc

        if response is None or response.status >= 400:
            status = response.status if response else "no response"
            raise UfcStatsFetchError(f"navigation to {url} returned HTTP {status}")

        self._page.wait_for_timeout(wait_ms)
        return self._page.content()
