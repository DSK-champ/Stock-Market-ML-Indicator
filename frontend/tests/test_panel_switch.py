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

    # 1. Initial Live Panel State
    badge = page.locator("#pred-source-badge").text_content().strip()
    print("Initial Source Badge:", badge)
    assert badge == "LIVE", f"Expected LIVE, got {badge}"

    live_time = page.locator("#pred-candle-time").text_content().strip()
    print("Initial Live Time:", live_time)

    # 2. Switch to Replay Tab
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(1000)

    replay_badge = page.locator("#pred-source-badge").text_content().strip()
    print("After Switch to Replay, Source Badge:", replay_badge)
    assert replay_badge == "REPLAY", f"Expected REPLAY, got {replay_badge}"

    # Step in replay
    page.locator("#btn-step-next").click()
    page.wait_for_timeout(400)
    replay_time = page.locator("#pred-candle-time").text_content().strip()
    print("Replay Stepped Time:", replay_time)

    # 3. Switch back to Live Tab
    page.locator("#btn-tab-live").click()
    page.wait_for_timeout(800)

    switched_live_badge = page.locator("#pred-source-badge").text_content().strip()
    switched_live_time = page.locator("#pred-candle-time").text_content().strip()
    print("After switching back to Live, Badge:", switched_live_badge, "Time:", switched_live_time)

    assert switched_live_badge == "LIVE", f"Expected LIVE, got {switched_live_badge}"
    assert switched_live_time == live_time, f"Expected live time {live_time}, got {switched_live_time}"

    # 4. Click a log table row to verify interactive inspection
    second_row = page.locator("#log-table-body tr").nth(3)
    row_time = second_row.locator("td").first.text_content().strip()
    second_row.click()
    page.wait_for_timeout(400)

    inspected_time = page.locator("#pred-candle-time").text_content().strip()
    print("Clicked row time:", row_time, "Panel time:", inspected_time)
    assert row_time[:5] in inspected_time, f"Row inspection mismatch: {row_time} vs {inspected_time}"

    print("ALL PREDICTION PANEL SWITCHING TESTS PASSED!")
    browser.close()
