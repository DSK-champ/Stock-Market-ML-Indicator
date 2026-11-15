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
    page.wait_for_timeout(4000)

    # Let's inspect the indicator study data
    res = page.evaluate("""() => {
        const chart = window.tvWidget.activeChart();
        const study = chart.getAllStudies().find(s => s.name.includes("2401CS45"));
        if (!study) return "study not found";
        
        // We can inspect the plots
        return {
            id: study.id,
            name: study.name,
            plots: study
        };
    }""")
    print("Study result:", res)
    browser.close()
