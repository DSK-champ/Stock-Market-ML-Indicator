"""Verification script for index.html Live Prediction Panel and 2401CS45 Indicator."""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

FRONTEND_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML_URI = f"file:///{ (FRONTEND_DIR / 'index.html').resolve().as_posix() }"
ARTIFACTS_DIR = Path(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949")

def test_index_live_prediction_panel():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files", "--disable-web-security"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        
        # Log browser console and errors
        page.on("console", lambda msg: print(f"[CONSOLE {msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[PAGE ERROR] {err}"))
        page.on("dialog", lambda dialog: dialog.accept())

        print(f"Navigating to {INDEX_HTML_URI}...")
        page.goto(INDEX_HTML_URI)
        page.wait_for_timeout(3000)

        # 1. Verify TradingView Chart and Indicator
        chart_container = page.locator("#tv_chart_container")
        assert chart_container.is_visible(), "TradingView chart container should be visible"
        print("✓ TradingView chart container is visible")

        # 2. Check Custom Panel Toggle Button
        toggle_btn = page.locator("#custom-panel-toggle")
        assert toggle_btn.is_visible(), "Custom panel toggle button should be visible"
        assert toggle_btn.get_attribute("aria-expanded") == "false", "Initial toggle should be false"
        print("✓ Custom panel toggle button present and initially closed")

        # 3. Click toggle button to open prediction panel
        toggle_btn.click()
        page.wait_for_timeout(2500)

        assert toggle_btn.get_attribute("aria-expanded") == "true", "Toggle should now be expanded"
        panel = page.locator("#custom-empty-panel")
        assert panel.is_visible(), "Prediction panel should be visible after toggle click"
        print("✓ Prediction panel successfully opened")

        # 4. Verify STRICTLY LIVE: No Replay tab and No Features tab
        replay_tab_count = page.locator("#custom-empty-panel #btn-tab-replay, #custom-empty-panel #log-tab-replay").count()
        features_tab_count = page.locator("#custom-empty-panel #log-tab-features, #custom-empty-panel #features-panel-view").count()
        assert replay_tab_count == 0, f"Expected 0 Replay tabs, found {replay_tab_count}"
        assert features_tab_count == 0, f"Expected 0 Features tabs, found {features_tab_count}"
        print("✓ Verified strictly Live: Zero Replay tabs and Zero Features tabs")

        # 5. Verify Live Prediction Panel components
        pred_source = page.locator("#pred-source-badge").text_content().strip()
        print("Live source badge:", pred_source)
        assert pred_source == "LIVE", f"Expected LIVE, got {pred_source}"

        market_tag = page.locator("#pred-market-badge").text_content().strip()
        trend_tag = page.locator("#pred-trend-badge").text_content().strip()
        phase_tag = page.locator("#pred-phase-badge").text_content().strip()
        action_tag = page.locator("#pred-price-action-badge").text_content().strip()
        print(f"Tags: Market={market_tag}, Trend={trend_tag}, Phase={phase_tag}, Action={action_tag}")
        assert market_tag in ["BULLISH", "BEARISH"], f"Unexpected market tag {market_tag}"
        assert trend_tag in ["UPTREND", "DOWNTREND"], f"Unexpected trend tag {trend_tag}"
        assert phase_tag in ["IMPULSE", "BREAKTHROUGH", "PULLBACK"], f"Unexpected phase tag {phase_tag}"
        assert action_tag in ["UP", "DOWN", "NEUTRAL"], f"Unexpected action tag {action_tag}"

        prob_down = page.locator("#pred-prob-down").text_content().strip()
        prob_neu = page.locator("#pred-prob-neu").text_content().strip()
        prob_up = page.locator("#pred-prob-up").text_content().strip()
        print(f"Probabilities: DOWN={prob_down}, NEUTRAL={prob_neu}, UP={prob_up}")
        assert "%" in prob_down and "%" in prob_neu and "%" in prob_up

        final_badge = page.locator("#pred-final-badge").text_content().strip()
        conf = page.locator("#pred-confidence").text_content().strip()
        print(f"Decision: {final_badge}, Confidence: {conf}")
        assert final_badge in ["UP", "DOWN", "NEUTRAL"]
        assert "Confidence" in conf

        smart_price = page.locator("#pred-smart-price").text_content().strip()
        upper_cap = page.locator("#pred-upper-cap").text_content().strip()
        lower_cap = page.locator("#pred-lower-cap").text_content().strip()
        print(f"Targets: Smart={smart_price}, Upper={upper_cap}, Lower={lower_cap}")
        assert "$" in smart_price and "$" in upper_cap and "$" in lower_cap

        # 6. Verify Live Prediction Log Table
        log_rows = page.locator("#log-table-body tr")
        row_count = log_rows.count()
        print(f"Log table rows count: {row_count}")
        assert row_count > 0, "Log table should have rows populated"

        # Verify first row headers/columns
        first_row_cells = log_rows.first.locator("td")
        assert first_row_cells.count() == 8, f"Expected 8 columns, got {first_row_cells.count()}"
        print("First row values:", [first_row_cells.nth(i).text_content().strip() for i in range(8)])

        # 7. Verify Accuracy, Cov, and Calls Metrics Display are Removed
        metrics_display_count = page.locator("#log-stat-calls, #log-stat-accuracy, #log-stat-coverage, #log-summary-strip").count()
        assert metrics_display_count == 0, f"Expected 0 metrics display elements, found {metrics_display_count}"
        print("✓ Verified accuracy, cov, and calls metrics display are completely removed from index prediction panel")

        # 7B. Strictly Zero Symbols and Zero Dashes Verification
        panel_text = panel.inner_text()
        for forbidden in ["✓", "✗", "—", "–"]:
            assert forbidden not in panel_text, f"Forbidden symbol or dash '{forbidden}' found in panel text!"
        print("✓ Verified strictly zero symbols (✓, ✗) and zero dashes (—, –) in entire panel text")

        # Verify all verdicts are strictly CORRECT, WRONG, or PENDING
        verdict_elements = page.locator("#log-table-body .verdict-mark").all_text_contents()
        for v in verdict_elements:
            assert v in ["CORRECT", "WRONG", "PENDING"], f"Invalid verdict badge '{v}' found!"
        print(f"✓ All {len(verdict_elements)} verdict marks are strictly CORRECT, WRONG, or PENDING")
        correct_pill = page.locator("#custom-empty-panel .filter-pill[data-filter='correct']")
        correct_pill.click()
        page.wait_for_timeout(300)
        filtered_correct_count = page.locator("#log-table-body tr").count()
        print(f"Correct filtered rows: {filtered_correct_count}")

        all_pill = page.locator("#custom-empty-panel .filter-pill[data-filter='all']")
        all_pill.click()
        page.wait_for_timeout(300)
        assert page.locator("#log-table-body tr").count() == row_count

        # 8B. Verify Horizontal Sideways Scrolling on Logs Table
        scroll_info = page.evaluate("""() => {
            const c = document.getElementById("log-table-container");
            const canScroll = c.scrollWidth > c.clientWidth;
            c.scrollLeft = 180;
            return {
                canScroll,
                scrollWidth: c.scrollWidth,
                clientWidth: c.clientWidth,
                scrolledLeft: c.scrollLeft
            };
        }""")
        print("Logs table scroll metrics:", scroll_info)
        assert scroll_info["canScroll"], f"Logs table must be horizontally scrollable ({scroll_info['scrollWidth']} > {scroll_info['clientWidth']})"
        assert scroll_info["scrolledLeft"] > 0, "scrollLeft should be greater than 0 after scrolling sideways"
        print("✓ Verified logs table sideways horizontal scrolling")

        # 9. Test Click Row Inspection
        if row_count > 3:
            row_3 = log_rows.nth(3)
            row_3_time = row_3.locator("td").first.text_content().strip()
            row_3.click()
            page.wait_for_timeout(300)
            inspected_time = page.locator("#pred-candle-time").text_content().strip()
            print(f"Clicked row 3 time: {row_3_time}, Inspected panel time: {inspected_time}")
            assert row_3_time[:5] in inspected_time

        # 10. Capture Screenshots
        screenshot_full = ARTIFACTS_DIR / "index_live_prediction_panel_full.png"
        screenshot_panel = ARTIFACTS_DIR / "index_live_prediction_panel_zoom.png"
        page.screenshot(path=str(screenshot_full))
        panel.screenshot(path=str(screenshot_panel))
        print(f"✓ Saved full screenshot: {screenshot_full}")
        print(f"✓ Saved panel screenshot: {screenshot_panel}")

        # 11. Verify 2401CS45 indicator study in TradingView
        indicator_check = page.evaluate("""async () => {
            if (window.tvWidget && typeof window.tvWidget.activeChart === 'function') {
                const chart = window.tvWidget.activeChart();
                const studies = chart.getAllStudies();
                const study2401 = studies.find(s => s.name && s.name.includes("2401CS45"));
                return {
                    studies: studies.map(s => s.name),
                    has2401: !!study2401
                };
            }
            return { studies: [], has2401: false };
        }""")
        print("Active chart studies:", indicator_check)
        assert indicator_check["has2401"], "Indicator 2401CS45 should be active on the chart"

        # Also capture chart container screenshot
        screenshot_chart = ARTIFACTS_DIR / "index_chart_2401cs45_indicator.png"
        chart_container.screenshot(path=str(screenshot_chart))
        print(f"✓ Saved chart screenshot: {screenshot_chart}")

        browser.close()
        print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_index_live_prediction_panel()
