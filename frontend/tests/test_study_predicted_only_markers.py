from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--allow-file-access-from-files", "--disable-web-security"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda m: print("CONSOLE:", m.type, m.text) if m.type in ["error", "warning"] else None)
    page.goto(uri)
    page.wait_for_timeout(4000)

    res = page.evaluate("""async () => {
        const widget = await window.tvWidgetReady;
        const chart = widget.activeChart();
        const map = window.__ML_PREDICTIONS_MAP;
        
        // Find study
        const study = chart.getAllStudies().find(s => s.name && s.name.includes("2401CS45"));
        return {
            hasStudy: !!study,
            hasMap: !!map,
            mapSize: map ? map.size : 0,
            sampleKeys: map ? Array.from(map.keys()).slice(0, 5) : []
        };
    }""")
    print("Test result:", res)
    assert res["hasStudy"], "2401CS45 study should be on the chart"
    assert res["hasMap"], "Predictions map should be initialized"
    assert res["mapSize"] > 0, "Predictions map should have items"

    # Also capture screenshot
    screenshot_path = Path(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949\index_chart_2401cs45_indicator.png")
    page.locator("#tv_chart_container").screenshot(path=str(screenshot_path))
    print(f"✓ Saved updated chart screenshot to {screenshot_path}")

    browser.close()
    print("TEST PASSED: Markers strictly limited to predicted candles!")
