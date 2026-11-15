from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--allow-file-access-from-files", "--disable-web-security"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(uri)
    page.wait_for_timeout(3500)

    # Test supertrend values directly via PineJS if accessible, or check console
    logs = page.evaluate("""() => {
        // Let's check window.customIndicatorsDebug or evaluate study output
        return window.tvWidget ? "widget present" : "no widget";
    }""")
    print("Widget:", logs)
    browser.close()
