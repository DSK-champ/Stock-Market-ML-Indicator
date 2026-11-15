from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--allow-file-access-from-files", "--disable-web-security"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: print(f"[{m.type}] {m.text}"))
    
    # Set a global test variable before chart loads
    page.add_init_script("window.__TEST_GLOBAL__ = 'HELLO_FROM_WINDOW';")
    
    page.goto(uri)
    page.wait_for_timeout(3500)
    
    res = page.evaluate("""() => {
        return window.__TEST_GLOBAL__;
    }""")
    print("Window test global:", res)
    browser.close()
