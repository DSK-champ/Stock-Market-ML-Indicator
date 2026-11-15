from pathlib import Path
from playwright.sync_api import sync_playwright

p_dir = Path("frontend")
uri = f"file:///{ (p_dir / 'blank.html').resolve().as_posix() }?mock=1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(uri)
    page.wait_for_timeout(2000)

    # Click on replay view
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(1000)

    # Step forward 12 steps to show historical prediction rays, actual line, and predicted line
    for _ in range(12):
        page.locator("#btn-step-next").click()
        page.wait_for_timeout(250)

    page.wait_for_timeout(1000)
    out_path = Path("frontend/test-results/chart_smart_lines_verified.png")
    page.screenshot(path=str(out_path))
    print(f"Saved verified screenshot to {out_path}")
    browser.close()

