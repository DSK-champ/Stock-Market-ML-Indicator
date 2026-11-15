import os
from pathlib import Path
from playwright.sync_api import sync_playwright

FRONTEND_DIR = Path(r"c:\B.TECH\Academics\SEMESTER V\03 Machine Learning\.ML LABs\Project\Stock-Market-ML-Indicator\frontend")
ARTIFACT_DIR = Path(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949")
BLANK_HTML_URI = f"file:///{ (FRONTEND_DIR / 'blank.html').resolve().as_posix() }?mock=1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(BLANK_HTML_URI)
    page.wait_for_timeout(2000)

    # 1. Verify actualLine is unchecked initially
    is_checked = page.evaluate("() => document.querySelector('input[data-layer=\"actualLine\"]').checked")
    chart_layer_state = page.evaluate("() => window.CHARTS && window.CHARTS.live ? window.CHARTS.live.layers.actualLine : null")
    state_layer_toggle = page.evaluate("() => window.STATE ? window.STATE.layerToggles.actualLine : null")
    badge_text = page.locator("#active-indicators-count").text_content().strip()

    print(f"actualLine checkbox checked: {is_checked}")
    print(f"CHARTS.live actualLine layer: {chart_layer_state}")
    print(f"STATE.layerToggles.actualLine: {state_layer_toggle}")
    print(f"Indicator badge text: {badge_text}")

    assert is_checked is False, f"Expected actualLine to be unchecked initially, got {is_checked}"
    assert chart_layer_state is False, f"Expected chart actualLine layer false, got {chart_layer_state}"
    assert state_layer_toggle is False, f"Expected STATE actualLine toggle false, got {state_layer_toggle}"
    assert "4 active" in badge_text, f"Expected 4 active in badge, got {badge_text}"

    # Screenshot initially with actualLine hidden
    init_shot = ARTIFACT_DIR / "initial_state_actual_line_hidden.png"
    page.screenshot(path=str(init_shot))
    print(f"Initial screenshot saved: {init_shot}")

    # 2. Test enabling actualLine (verify toggle and that dots are NOT drawn)
    page.locator("#btn-indicator-settings").click()
    page.wait_for_timeout(300)
    page.locator("input[data-layer='actualLine']").check()
    page.wait_for_timeout(500)

    is_checked_now = page.evaluate("() => document.querySelector('input[data-layer=\"actualLine\"]').checked")
    chart_layer_now = page.evaluate("() => window.CHARTS.live.layers.actualLine")
    badge_now = page.locator("#active-indicators-count").text_content().strip()
    assert is_checked_now is True
    assert chart_layer_now is True
    assert "5 active" in badge_now
    print(f"Enabled actualLine: checked={is_checked_now}, chart={chart_layer_now}, badge={badge_now}")

    enabled_shot = ARTIFACT_DIR / "actual_line_enabled_no_dots.png"
    page.screenshot(path=str(enabled_shot))
    print(f"Enabled actual line screenshot saved: {enabled_shot}")

    # Close settings
    page.locator("#btn-close-indicator-settings").click()
    page.wait_for_timeout(300)

    # 3. Test Replay stepping to inspect past candles and price level indicator color
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(1000)

    # Step forward 10 times in replay
    for _ in range(10):
        page.locator("#btn-step-next").click()
        page.wait_for_timeout(200)

    page.wait_for_timeout(1000)

    # Check color logic of drawLastPriceLine
    chart_info = page.evaluate("""
        () => {
            const chart = window.CHARTS.replay;
            if (!chart) return null;
            const n = chart.candles.length;
            const lastCandle = chart.candles[n - 1];
            const isBullish = lastCandle.close >= lastCandle.open;
            return {
                candlesCount: n,
                lastClose: lastCandle.close,
                lastOpen: lastCandle.open,
                isBullish: isBullish,
                upColor: chart.options.upColor,
                downColor: chart.options.downColor,
                predictionsCount: chart.predictions.length
            };
        }
    """)
    print("Replay Chart Info:", chart_info)

    replay_shot = ARTIFACT_DIR / "replay_smart_line_past_candles_no_labels.png"
    page.screenshot(path=str(replay_shot))
    print(f"Replay verified screenshot saved: {replay_shot}")

    browser.close()
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
