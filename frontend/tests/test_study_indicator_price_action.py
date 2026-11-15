import time
from pathlib import Path
from playwright.sync_api import sync_playwright

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--allow-file-access-from-files", "--disable-web-security"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: print(f"[{m.type}] {m.text}"))
    page.on("pageerror", lambda e: print(f"[PAGE ERROR] {e}"))

    print(f"Loading {uri}...")
    page.goto(uri)
    page.wait_for_timeout(4000)

    # Check chart and studies
    studies = page.evaluate("""() => {
        if (window.tvWidget && typeof window.tvWidget.activeChart === 'function') {
            const chart = window.tvWidget.activeChart();
            return chart.getAllStudies().map(s => ({ id: s.id, name: s.name }));
        }
        return [];
    }""")
    print("Studies on chart:", studies)

    browser.close()
