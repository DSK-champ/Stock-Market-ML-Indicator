from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files", "--disable-web-security"])
    page = browser.new_page()
    page.goto(uri)
    page.wait_for_timeout(3000)

    res = page.evaluate("""() => {
        return typeof window !== 'undefined';
    }""")
    print("Page evaluated:", res)
    browser.close()
