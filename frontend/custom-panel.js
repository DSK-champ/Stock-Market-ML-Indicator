(() => {
  const fallbackPanelTop = 48;
  const fallbackToggleTop = 176;

  function visible(element, view) {
    const rect = element.getBoundingClientRect();
    const style = view.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }

  function getChartFrames() {
    return [...document.querySelectorAll("#tv_chart_container iframe")];
  }

  function inspectFrame(frame) {
    try {
      const doc = frame.contentDocument;
      const view = frame.contentWindow;
      if (!doc || !view) return null;

      const buttons = [...doc.querySelectorAll("button, [role='button'], [role='tab']")]
        .filter((item) => visible(item, view));
      function getBottom(labels) {
        const matches = buttons.filter((item) => labels.test([
          item.getAttribute("aria-label"),
          item.getAttribute("title"),
          item.textContent,
        ].filter(Boolean).join(" ")));
        return matches.length ? Math.max(...matches.map((item) => item.getBoundingClientRect().bottom)) : null;
      }

      const frameRect = frame.getBoundingClientRect();
      return {
        panelTop: getBottom(/save|settings|chart properties|full\s*screen|snapshot|camera/i),
        toggleTop: getBottom(/watchlist|details|data\s*window|object\s*tree/i),
        frameTop: frameRect.top,
      };
    } catch (_error) {
      // The chart may move its UI into a frame that does not expose its document.
      return null;
    }
  }

  function initialize() {
    const button = document.getElementById("custom-panel-toggle");
    const blankPageButton = document.getElementById("blank-page-button");
    const panel = document.getElementById("custom-empty-panel");
    const resizeHandle = document.getElementById("custom-panel-resize-handle");
    if (!button || !blankPageButton || !panel || !resizeHandle) return;

    const minWidth = 250;
    let resizedWidth = null;

    function getMaximumWidth() {
      return Math.max(minWidth, window.innerWidth - 40);
    }

    function setPanelWidth(width) {
      resizedWidth = Math.max(minWidth, Math.min(getMaximumWidth(), width));
      document.documentElement.style.setProperty("--right-col-w", `${resizedWidth}px`);
    }

    function startResize(event, edge) {
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = panel.getBoundingClientRect().width;
      const frames = getChartFrames();
      const previousPointerEvents = frames.map((frame) => frame.style.pointerEvents);
      frames.forEach((frame) => { frame.style.pointerEvents = "none"; });
      document.body.style.cursor = "ew-resize";
      document.body.style.userSelect = "none";

      function resize(moveEvent) {
        const delta = edge === "left" ? startX - moveEvent.clientX : moveEvent.clientX - startX;
        setPanelWidth(startWidth + delta);
      }

      function stopResize() {
        window.removeEventListener("pointermove", resize);
        window.removeEventListener("pointerup", stopResize);
        window.removeEventListener("pointercancel", stopResize);
        window.removeEventListener("blur", stopResize);
        frames.forEach((frame, index) => { frame.style.pointerEvents = previousPointerEvents[index]; });
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }

      window.addEventListener("pointermove", resize);
      window.addEventListener("pointerup", stopResize);
      window.addEventListener("pointercancel", stopResize);
      window.addEventListener("blur", stopResize);
    }

    resizeHandle.addEventListener("pointerdown", (event) => startResize(event, "left"));

    function resizeWithKeyboard(event, direction) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const currentWidth = resizedWidth ?? panel.getBoundingClientRect().width;
      const adjustment = (event.key === "ArrowLeft" ? 10 : -10) * direction;
      setPanelWidth(currentWidth + adjustment);
    }

    resizeHandle.addEventListener("keydown", (event) => resizeWithKeyboard(event, 1));

    button.addEventListener("click", async () => {
      const isOpen = button.getAttribute("aria-expanded") === "true";
      if (isOpen) {
        panel.hidden = true;
        button.setAttribute("aria-expanded", "false");
        window.predictionPanelOpen = false;
        return;
      }

      button.disabled = true;
      try {
        if (typeof window.preparePredictionChart === "function") {
          await window.preparePredictionChart();
        }
        panel.hidden = false;
        button.setAttribute("aria-expanded", "true");
        window.predictionPanelOpen = true;
      } finally {
        button.disabled = false;
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) {
        panel.hidden = true;
        button.setAttribute("aria-expanded", "false");
        window.predictionPanelOpen = false;
      }
    });

    function syncToChartPanel() {
      let detected = null;
      for (const frame of getChartFrames()) {
        detected = inspectFrame(frame);
        if (detected) break;
      }

      if (resizedWidth !== null) setPanelWidth(resizedWidth);
      const panelTop = detected?.panelTop === null || detected?.panelTop === undefined
        ? fallbackPanelTop
        : Math.max(0, Math.min(detected.frameTop + detected.panelTop + 4, window.innerHeight - 44));
      const toggleTop = detected?.toggleTop === null || detected?.toggleTop === undefined
        ? fallbackToggleTop
        : Math.max(0, Math.min(detected.frameTop + detected.toggleTop + 4, window.innerHeight - 44));
      panel.style.top = `${panelTop}px`;
      button.style.top = `${toggleTop}px`;
      blankPageButton.style.top = `${toggleTop + 40}px`;
    }

    window.addEventListener("resize", syncToChartPanel);
    syncToChartPanel();
    window.setInterval(syncToChartPanel, 1000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
