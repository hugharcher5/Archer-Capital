"""
Keep the Streamlit Community Cloud frontend awake.

Cloud puts an app to sleep after a period with no visitors, and a plain HTTP
GET does not count as a visit: the app only wakes when a real browser opens
its websocket session. So this opens the app in headless Chromium, clicks the
"Yes, get this app back up!" button if the sleep page is showing, and waits
for the app itself to render.

Run by .github/workflows/keep_awake.yml. Requires the app to be public;
a login wall is reported as a failure so it shows up red in Actions.
"""

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://archer-capital-35mdgw2nzvbnouvs8ug9sq.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Every visitor, public app or not, is bounced through share.streamlit.io's
        # auth handshake first, so judge by what finally renders, not the URL
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

        wake = page.get_by_role("button", name=WAKE_BUTTON)
        no_access = page.get_by_text("You do not have access to this app")
        app_frame = page.frame_locator("iframe").first.locator("[data-testid='stApp']")
        for _ in range(60):
            if wake.is_visible() or no_access.is_visible() or app_frame.count():
                break
            page.wait_for_timeout(1_000)

        if no_access.is_visible():
            print("[keep_awake] Login wall: make the app public in Streamlit Cloud settings.")
            return 1
        if wake.is_visible():
            print("[keep_awake] App was asleep; clicking wake button.")
            wake.click()
        else:
            print("[keep_awake] App was already awake.")

        # The app runs inside an iframe on *.streamlit.app; wait for Streamlit's root to render
        try:
            page.frame_locator("iframe").first.locator("[data-testid='stApp']").wait_for(timeout=180_000)
        except PlaywrightTimeout:
            page.locator("[data-testid='stApp']").wait_for(timeout=30_000)
        print("[keep_awake] App is up.")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
