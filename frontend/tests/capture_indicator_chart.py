import time
from pathlib import Path
from playwright.sync_api import sync_playwright

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"
artifacts_dir = Path(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949")

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--allow-file-access-from-files", "--disable-web-security"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(uri)
    page.wait_for_timeout(4500)

    # Take screenshot of the chart with 2401CS45 study
    chart_screenshot = artifacts_dir / "index_chart_2401cs45_indicator.png"
    page.locator("#tv_chart_container").screenshot(path=str(chart_screenshot))
    print("Saved chart screenshot to:", chart_screenshot)

    browser.close()
