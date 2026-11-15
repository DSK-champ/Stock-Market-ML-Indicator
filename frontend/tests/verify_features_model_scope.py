import os
from pathlib import Path
from playwright.sync_api import sync_playwright

FRONTEND_DIR = Path(r"c:\B.TECH\Academics\SEMESTER V\03 Machine Learning\.ML LABs\Project\Stock-Market-ML-Indicator\frontend")
ARTIFACT_DIR = Path(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949")
BLANK_HTML_URI = f"file:///{ (FRONTEND_DIR / 'blank.html').resolve().as_posix() }?mock=1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})

    # Collect console logs/errors
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    page.goto(BLANK_HTML_URI)
    page.wait_for_timeout(1500)

    # 1. Switch to Features tab
    page.locator("#log-tab-features").click()
    page.wait_for_timeout(500)

    # Verify Features View is visible
    features_view = page.locator("#features-panel-view")
    assert features_view.is_visible()

    # Verify Horizon-2 Features (101) active by default
    active_model = page.locator("#features-model-pills .filter-pill.active").text_content().strip()
    assert "Horizon-2" in active_model or "101" in active_model
    rows_h2 = page.locator("#features-table-body tr")
    count_h2 = rows_h2.count()
    print(f"Horizon-2 rows count: {count_h2}")
    assert count_h2 == 101, f"Expected 101 rows for Horizon-2, got {count_h2}"

    # Verify table header has Active Models
    headers = page.evaluate("() => Array.from(document.querySelectorAll('.features-table th')).map(th => th.textContent.trim())")
    print(f"Headers: {headers}")
    assert "Active Models" in headers or "ACTIVE MODELS" in [h.upper() for h in headers]

    # Save screenshot of Horizon-2 view
    shot_h2 = ARTIFACT_DIR / "features_h2_dedicated_101.png"
    page.screenshot(path=str(shot_h2))
    print(f"Saved: {shot_h2}")

    # 2. Test Returns filter (should only be feat_ret_1 and feat_ret_2)
    page.locator("#features-category-pills button[data-cat='returns']").click()
    page.wait_for_timeout(300)
    ret_rows = page.locator("#features-table-body tr")
    count_ret = ret_rows.count()
    print(f"Returns count: {count_ret}")
    assert count_ret == 2, f"Expected exactly 2 return features (feat_ret_1, feat_ret_2), got {count_ret}"

    shot_ret = ARTIFACT_DIR / "features_h2_returns_category.png"
    page.screenshot(path=str(shot_ret))
    print(f"Saved: {shot_ret}")

    # Reset to All Groups
    page.locator("#features-category-pills button[data-cat='all']").click()
    page.wait_for_timeout(300)

    # 3. Click Linear LR (41)
    page.locator("#features-model-pills button[data-model='linear']").click()
    page.wait_for_timeout(400)

    rows_linear = page.locator("#features-table-body tr")
    count_linear = rows_linear.count()
    print(f"Linear LR rows count: {count_linear}")
    assert count_linear == 41, f"Expected 41 rows for Linear LR, got {count_linear}"

    counter_linear = page.locator(".features-count-label").text_content().strip()
    print(f"Linear counter label: {counter_linear}")
    assert "41" in counter_linear

    shot_linear = ARTIFACT_DIR / "features_model_scope_linear_41.png"
    page.screenshot(path=str(shot_linear))
    print(f"Saved: {shot_linear}")

    # 4. Click Meta-Model (15)
    page.locator("#features-model-pills button[data-model='meta']").click()
    page.wait_for_timeout(400)

    rows_meta = page.locator("#features-table-body tr")
    count_meta = rows_meta.count()
    print(f"Meta-Model rows count: {count_meta}")
    assert count_meta == 15, f"Expected 15 rows for Meta-Model, got {count_meta}"

    counter_meta = page.locator(".features-count-label").text_content().strip()
    print(f"Meta-Model counter label: {counter_meta}")
    assert "15" in counter_meta

    shot_meta = ARTIFACT_DIR / "features_model_scope_meta_15.png"
    page.screenshot(path=str(shot_meta))
    print(f"Saved: {shot_meta}")

    # 5. Scroll horizontally to verify Active Models badges and formula
    page.locator("#features-model-pills button[data-model='all']").click()
    page.wait_for_timeout(300)
    page.evaluate("() => { const c = document.getElementById('features-table-container'); if (c) c.scrollLeft = 320; }")
    page.wait_for_timeout(300)

    shot_scrolled = ARTIFACT_DIR / "features_table_scrolled_models.png"
    page.screenshot(path=str(shot_scrolled))
    print(f"Saved: {shot_scrolled}")

    # Check for console errors
    print(f"Console errors: {errors}")
    assert len(errors) == 0, f"Encountered console errors: {errors}"

    browser.close()
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
