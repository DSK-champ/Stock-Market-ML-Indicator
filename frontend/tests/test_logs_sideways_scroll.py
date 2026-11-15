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
    page.wait_for_timeout(3000)

    # 1. Open prediction panel
    toggle_btn = page.locator("#custom-panel-toggle")
    toggle_btn.click()
    page.wait_for_timeout(2000)

    container = page.locator("#log-table-container")
    assert container.is_visible(), "Table container should be visible"

    # 2. Check scroll metrics
    metrics_initial = page.evaluate("""() => {
        const c = document.getElementById("log-table-container");
        return {
            clientWidth: c.clientWidth,
            scrollWidth: c.scrollWidth,
            scrollLeft: c.scrollLeft,
            canScrollHorizontal: c.scrollWidth > c.clientWidth
        };
    }""")
    print("Initial Table Container Metrics:", metrics_initial)
    assert metrics_initial["canScrollHorizontal"], f"Container should be horizontally scrollable: scrollWidth ({metrics_initial['scrollWidth']}) > clientWidth ({metrics_initial['clientWidth']})"
    assert metrics_initial["scrollLeft"] == 0

    # 3. Take screenshot at scrollLeft = 0
    page.locator("#custom-empty-panel").screenshot(path=str(artifacts_dir / "logs_table_scrolled_left.png"))

    # 4. Scroll sideways (horizontally)
    page.evaluate("""() => {
        const c = document.getElementById("log-table-container");
        c.scrollLeft = 200;
    }""")
    page.wait_for_timeout(500)

    metrics_scrolled = page.evaluate("""() => {
        const c = document.getElementById("log-table-container");
        return {
            scrollLeft: c.scrollLeft
        };
    }""")
    print("After Sideways Scroll Metrics:", metrics_scrolled)
    assert metrics_scrolled["scrollLeft"] > 0, "scrollLeft should be greater than 0 after scrolling sideways"

    # 5. Take screenshot of horizontally scrolled table
    page.locator("#custom-empty-panel").screenshot(path=str(artifacts_dir / "logs_table_scrolled_right.png"))

    print("✓ Sideways scroll verified successfully!")
    browser.close()
