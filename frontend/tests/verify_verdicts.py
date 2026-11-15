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

    # 1. Check table headers
    headers = page.evaluate("() => Array.from(document.querySelectorAll('.log-table th')).map(th => th.textContent.trim())")
    print("Headers:", headers)
    assert headers[0].upper() == "TIME", f"Header 0 expected TIME, got {headers[0]}"
    assert headers[1].upper() == "PRICE ACTION", f"Header 1 expected PRICE ACTION, got {headers[1]}"
    assert headers[2].upper() == "VERDICT", f"Header 2 expected VERDICT, got {headers[2]}"

    # 2. Check rows in initial state
    rows_info = page.evaluate("""() => {
        const rows = Array.from(document.querySelectorAll('#log-table-body tr'));
        return rows.map((tr, idx) => {
            const tds = Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim());
            return { idx, time: tds[0], action: tds[1], verdict: tds[2] };
        });
    }""")

    print(f"Total initial rows loaded: {len(rows_info)}")
    for r in rows_info[:10]:
        print(f"Row {r['idx']}: Time={r['time']} Action={r['action']} Verdict={r['verdict']}")

    # Validation:
    assert len(rows_info) >= 3, "Expected at least 3 rows in mock"
    assert rows_info[0]["verdict"] == "—", f"Row 0 verdict must be '—', got {rows_info[0]['verdict']}"
    assert rows_info[1]["verdict"] == "—", f"Row 1 verdict must be '—', got {rows_info[1]['verdict']}"

    for r in rows_info[2:]:
        assert r["verdict"] in ["✓ CORRECT", "✗ WRONG"], f"Row {r['idx']} verdict not validated! Got: {r['verdict']}"

    # 3. Test real-time progression in Replay Mode
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(800)

    # Click Test now to simulate candles and predictions
    for step in range(5):
        page.locator("#btn-step-next").click()
        page.wait_for_timeout(300)

    replay_rows = page.evaluate("""() => {
        const rows = Array.from(document.querySelectorAll('#log-table-body tr'));
        return rows.map((tr, idx) => {
            const tds = Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim());
            return { idx, time: tds[0], action: tds[1], verdict: tds[2] };
        });
    }""")

    print(f"Replay rows count after 5 steps: {len(replay_rows)}")
    for r in replay_rows[:8]:
        print(f"Replay Row {r['idx']}: Time={r['time']} Action={r['action']} Verdict={r['verdict']}")

    if len(replay_rows) >= 1:
        assert replay_rows[0]["verdict"] == "—", f"Replay Row 0 must be '—', got {replay_rows[0]['verdict']}"
    if len(replay_rows) >= 2:
        assert replay_rows[1]["verdict"] == "—", f"Replay Row 1 must be '—', got {replay_rows[1]['verdict']}"
    if len(replay_rows) >= 3:
        for r in replay_rows[2:]:
            assert r["verdict"] in ["✓ CORRECT", "✗ WRONG"], f"Replay Row {r['idx']} not validated! Got {r['verdict']}"

    # Screenshot
    res_path = Path("frontend/test-results/verdicts_validation_verified.png")
    page.screenshot(path=str(res_path))
    print(f"Saved screenshot to {res_path}")

    print("ALL VERDICT RULES FULLY VERIFIED AND VALIDATED!")
    browser.close()
