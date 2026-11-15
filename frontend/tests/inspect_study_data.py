from playwright.sync_api import sync_playwright
from pathlib import Path

uri = f"file:///{Path('frontend/index.html').resolve().as_posix()}"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--allow-file-access-from-files", "--disable-web-security"])
    page = browser.new_page()
    page.goto(uri)
    page.wait_for_timeout(3500)

    res = page.evaluate("""async () => {
        const inds = await window.getCustomIndicators();
        const ind = inds.find(i => i.name === '2401CS45');
        const instance = new ind.constructor();
        
        // Mock PineJS context for an unpredicted candle (e.g. 5 days ago: time = 1000000000)
        let unpredictedMarkerResult = null;
        let mockContextUnpredicted = {
            new_var: (x) => ({ get: () => x, set: () => {} }),
            symbol: { time: 1000000000 * 1000, isNewBar: true, isBarClosed: true }
        };
        
        // Call main for an unpredicted bar
        const unpredictedOut = instance.main({
            new_var: (x) => ({ get: () => x, set: () => {} }),
            symbol: { time: 1000000000 * 1000 }
        }, (idx) => (idx === 0 ? 20 : 2.0));
        
        // Find a predicted time
        const predKeys = Array.from(window.__ML_PREDICTIONS_MAP.keys());
        const predTime = predKeys[0];
        const predictedOut = instance.main({
            new_var: (x) => ({ get: () => x, set: () => {} }),
            symbol: { time: predTime * 1000 }
        }, (idx) => (idx === 0 ? 20 : 2.0));

        return {
            unpredictedOut: {
                baseline: !isNaN(unpredictedOut[0]),
                upper: !isNaN(unpredictedOut[1]),
                lower: !isNaN(unpredictedOut[2]),
                plottedPath: unpredictedOut[3],
                upMarker: unpredictedOut[4],
                downMarker: unpredictedOut[5],
                neuMarker: unpredictedOut[6]
            },
            predictedOut: {
                baseline: !isNaN(predictedOut[0]),
                upper: !isNaN(predictedOut[1]),
                lower: !isNaN(predictedOut[2]),
                plottedPath: predictedOut[3],
                upMarker: predictedOut[4],
                downMarker: predictedOut[5],
                neuMarker: predictedOut[6]
            }
        };
    }""")
    print("Exact study plot outputs:")
    print("Unpredicted candle output:", res["unpredictedOut"])
    print("Predicted candle output:", res["predictedOut"])
    
    # Assertions
    import math
    assert math.isnan(res["unpredictedOut"]["upMarker"]), "Unpredicted candle must NOT have UP marker"
    assert math.isnan(res["unpredictedOut"]["downMarker"]), "Unpredicted candle must NOT have DOWN marker"
    assert math.isnan(res["unpredictedOut"]["neuMarker"]), "Unpredicted candle must NOT have NEUTRAL marker"
    assert math.isnan(res["unpredictedOut"]["plottedPath"]), "Unpredicted candle must NOT have plottedPath"
    
    # For predicted candle, at least one marker must be active (1) or plottedPath is a number
    has_active_marker = (res["predictedOut"]["upMarker"] == 1 or 
                         res["predictedOut"]["downMarker"] == 1 or 
                         res["predictedOut"]["neuMarker"] == 1)
    assert has_active_marker, "Predicted candle MUST have a marker"
    assert not math.isnan(res["predictedOut"]["plottedPath"]), "Predicted candle MUST have plottedPath"

    print("ALL STRICT MARKER RESTRICTIONS VERIFIED SUCCESSFULLY!")
    browser.close()
