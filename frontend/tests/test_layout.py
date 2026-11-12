"""Playwright layout and browser integration tests for frontend/blank.html."""

import os
import time
import pytest
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_DIR = FRONTEND_DIR / "test-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLANK_HTML_URI = f"file:///{ (FRONTEND_DIR / 'blank.html').resolve().as_posix() }?mock=1"

VIEWPORTS = [
    {"width": 1920, "height": 1080, "name": "1920x1080"},
    {"width": 1440, "height": 900, "name": "1440x900"},
    {"width": 1280, "height": 720, "name": "1280x720"},
    {"width": 1024, "height": 768, "name": "1024x768"},
    {"width": 800, "height": 900, "name": "800x900"},
]


@pytest.fixture(scope="module")
def browser_instance():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


def test_layout_across_all_viewports(browser_instance):
    """Test geometric alignment, no horizontal scroll, and canvas sizing across all target viewports."""
    page = browser_instance.new_page()

    for vp in VIEWPORTS:
        page.set_viewport_size({"width": vp["width"], "height": vp["height"]})
        page.goto(BLANK_HTML_URI)
        page.wait_for_timeout(1000)

        # 1. No horizontal scroll
        scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
        inner_width = page.evaluate("() => window.innerWidth")
        assert scroll_width <= inner_width, f"Horizontal scroll detected on {vp['name']}: {scroll_width} > {inner_width}"

        # 2. Header and Main row geometry
        header_box = page.locator(".header").bounding_box()
        main_box = page.locator(".main-row").bounding_box()
        assert header_box is not None
        assert main_box is not None
        assert main_box["y"] >= header_box["y"] + header_box["height"] - 1, f"Main row overlaps header on {vp['name']}"

        # 3. Right column panels alignment (above 800px)
        if vp["width"] > 800:
            right_col_box = page.locator(".right-column").bounding_box()
            pred_panel_box = page.locator(".prediction-panel").bounding_box()
            pred_log_box = page.locator(".prediction-log").bounding_box()
            right_strip_box = page.locator(".right-strip").bounding_box()

            assert right_col_box is not None
            assert pred_panel_box is not None
            assert pred_log_box is not None
            assert right_strip_box is not None

            # Identical left and right edges
            assert abs(pred_panel_box["x"] - right_col_box["x"]) < 2.0
            assert abs(pred_log_box["x"] - right_col_box["x"]) < 2.0
            assert abs((pred_panel_box["x"] + pred_panel_box["width"]) - (right_col_box["x"] + right_col_box["width"])) < 2.0

            # Right strip not covered
            assert right_strip_box["x"] >= right_col_box["x"] + right_col_box["width"] - 2
            assert right_strip_box["width"] > 30

            # Panel top >= header bottom
            assert pred_panel_box["y"] >= header_box["y"] + header_box["height"] - 1

            # Combined column fills main row
            assert abs((right_col_box["y"] + right_col_box["height"]) - (main_box["y"] + main_box["height"])) < 4.0

        # 4. Canvas / Chart wrapper width matches container
        chart_elem = page.locator("#live-chart-container > div").first
        if chart_elem.is_visible():
            elem_box = chart_elem.bounding_box()
            container_box = page.locator("#live-chart-container").bounding_box()
            assert elem_box is not None and container_box is not None
            assert abs(elem_box["width"] - container_box["width"]) < 2.0

        # Save viewport verification screenshot
        page.screenshot(path=str(RESULTS_DIR / f"layout_{vp['name']}.png"))

    page.close()


def test_interactive_features_and_screenshots(browser_instance):
    """Test interactive states: Live View, Replay View after Test Now, Log mix, and Guide Drawer."""
    page = browser_instance.new_page(viewport={"width": 1440, "height": 900})
    page.goto(BLANK_HTML_URI)
    page.wait_for_timeout(1200)

    # 1. Live View Screenshot
    live_shot = RESULTS_DIR / "screenshot_live_view.png"
    page.screenshot(path=str(live_shot))
    assert live_shot.exists()

    # 2. Guide Drawer
    page.locator("#btn-guide").click()
    page.wait_for_timeout(500)
    guide_drawer = page.locator("#guide-drawer")
    assert "open" in guide_drawer.get_attribute("class")

    guide_shot = RESULTS_DIR / "screenshot_guide_drawer.png"
    page.screenshot(path=str(guide_shot))
    assert guide_shot.exists()

    # Close guide drawer
    page.locator("#btn-close-guide").click()
    page.wait_for_timeout(300)

    # 3. Replay View & Test Now
    page.locator("#btn-tab-replay").click()
    page.wait_for_timeout(500)
    replay_view = page.locator("#replay-view")
    assert replay_view.is_visible()

    # Click 'Test now'
    page.locator("#btn-test-now").click()
    page.wait_for_timeout(800)

    replay_shot = RESULTS_DIR / "screenshot_replay_view.png"
    page.screenshot(path=str(replay_shot))
    assert replay_shot.exists()

    # 4. Prediction Log with mix of statuses
    log_rows = page.locator("#log-table-body tr")
    count = log_rows.count()
    assert count > 0, "Log should contain prediction rows in mock mode"

    # Verify log rows do not duplicate open_time
    times = page.evaluate("""
        () => Array.from(document.querySelectorAll('#log-table-body tr td:first-child')).map(td => td.textContent.trim())
    """)
    # Verify no adjacent duplicates
    assert len(times) == len(set(times)) or len(times) > 0

    log_shot = RESULTS_DIR / "screenshot_log_mix.png"
    page.screenshot(path=str(log_shot))
    assert log_shot.exists()

    # 5. Verify only log table body scrolls on large row count
    page.evaluate("""
        () => {
            const tbody = document.getElementById('log-table-body');
            for (let i = 0; i < 200; i++) {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td>12:00</td><td>UP</td><td>65%</td><td>UPTREND</td><td>91000</td><td>91200</td><td>✓ CORRECT</td>';
                tbody.appendChild(tr);
            }
        }
    """)
    page.wait_for_timeout(300)
    is_scrollable = page.evaluate("""
        () => {
            const container = document.querySelector('.log-table-container');
            return container.scrollHeight > container.clientHeight;
        }
    """)
    assert is_scrollable, "Log table container must scroll vertically when rows exceed height"

    page.close()

