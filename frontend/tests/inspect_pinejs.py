from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files", "--disable-web-security"])
    page = browser.new_page()
    page.on("console", lambda m: print("CONSOLE:", m.type, m.text))
    page.on("pageerror", lambda err: print("PAGE ERROR:", err))
    page.goto(uri)
    page.wait_for_timeout(3500)
    res = page.evaluate("""() => {
        return {
            hasMap: typeof window.__ML_PREDICTIONS_MAP !== 'undefined',
            mapVal: window.__ML_PREDICTIONS_MAP ? Array.from(window.__ML_PREDICTIONS_MAP.entries()).slice(0, 3) : null,
            tableRows: document.querySelectorAll("#log-table-body tr").length
        };
    }""")
    print("Result:", res)
    browser.close()
