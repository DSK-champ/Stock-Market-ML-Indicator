import time
from pathlib import Path
from playwright.sync_api import sync_playwright

p_dir = Path("frontend")
uri = f"file:///{ (p_dir / 'blank.html').resolve().as_posix() }?mock=1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(uri)
    page.wait_for_timeout(2000)

    # Switch to Replay tab
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(1000)

    # Check initial simulation candles length
    initial_len = page.evaluate("() => STATE.replaySimulationCandles.length")
    print(f"Initial replaySimulationCandles length: {initial_len}")
    assert initial_len >= 1000, f"Expected at least 1000 simulation candles, got {initial_len}"

    # Fast forward simulate 310 steps (past the 288-step / 24-hour mark)
    page.evaluate("""() => {
        for (let i = 0; i < 310; i++) {
            executeReplayStep();
        }
    }""")
    page.wait_for_timeout(1000)

    curr_idx = page.evaluate("() => STATE.replaySimulationIndex")
    badge_text = page.locator("#replay-date-badge").text_content()
    print(f"Current replaySimulationIndex: {curr_idx}")
    print(f"Date badge text: {badge_text}")

    assert curr_idx == 310, f"Expected index 310, got {curr_idx}"
    assert "Step 310" in badge_text, f"Badge did not reflect step 310: {badge_text}"

    # Verify that prediction rows and verdicts continue updating
    log_rows = page.evaluate("""() => {
        const rows = Array.from(document.querySelectorAll('#log-table-body tr'));
        return rows.length;
    }""")
    print(f"Total prediction rows in log: {log_rows}")
    assert log_rows > 300, f"Expected >300 prediction rows, got {log_rows}"

    print("SUCCESS: Replay smoothly continues past 24 hours (27th Aug 5:45 AM) through subsequent days!")
    browser.close()

