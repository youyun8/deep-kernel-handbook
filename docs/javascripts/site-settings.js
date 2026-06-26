(function () {
  const storageKey = "ml-perf-handbook-settings";
  const defaults = {
    density: "comfortable",
    width: "standard",
    codeWrap: false,
  };

  function loadSettings() {
    try {
      return {
        ...defaults,
        ...JSON.parse(localStorage.getItem(storageKey) || "{}"),
      };
    } catch (_error) {
      return { ...defaults };
    }
  }

  function saveSettings(settings) {
    localStorage.setItem(storageKey, JSON.stringify(settings));
  }

  function applySettings(settings) {
    const root = document.documentElement;
    root.dataset.readingDensity = settings.density;
    root.dataset.readingWidth = settings.width;
    root.dataset.codeWrap = String(Boolean(settings.codeWrap));
  }

  function button(label, active, onClick) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = `ml-setting-option${active ? " is-active" : ""}`;
    el.textContent = label;
    el.addEventListener("click", onClick);
    return el;
  }

  function renderPanel(settings, onChange) {
    const panel = document.createElement("aside");
    panel.className = "ml-settings-panel";
    panel.setAttribute("aria-label", "閱讀設定");
    panel.hidden = true;
    panel.innerHTML = `
      <div class="ml-settings-panel__head">
        <div>
          <p>設定</p>
          <strong>閱讀偏好</strong>
        </div>
        <button type="button" class="ml-settings-close" aria-label="關閉設定">×</button>
      </div>
      <div class="ml-settings-row" data-setting="density">
        <span>版面密度</span>
        <div class="ml-settings-options"></div>
      </div>
      <div class="ml-settings-row" data-setting="width">
        <span>內容寬度</span>
        <div class="ml-settings-options"></div>
      </div>
      <label class="ml-settings-check">
        <input type="checkbox" />
        <span>程式碼自動換行</span>
      </label>
    `;

    const density = panel.querySelector(
      '[data-setting="density"] .ml-settings-options'
    );
    density.append(
      button("舒適", settings.density === "comfortable", () =>
        onChange({ density: "comfortable" })
      ),
      button("緊湊", settings.density === "compact", () =>
        onChange({ density: "compact" })
      )
    );

    const width = panel.querySelector(
      '[data-setting="width"] .ml-settings-options'
    );
    width.append(
      button("標準", settings.width === "standard", () =>
        onChange({ width: "standard" })
      ),
      button("寬版", settings.width === "wide", () =>
        onChange({ width: "wide" })
      )
    );

    const check = panel.querySelector("input");
    check.checked = Boolean(settings.codeWrap);
    check.addEventListener("change", () =>
      onChange({ codeWrap: check.checked })
    );
    panel.querySelector(".ml-settings-close").addEventListener("click", () => {
      panel.hidden = true;
    });
    return panel;
  }

  function mount() {
    let settings = loadSettings();
    applySettings(settings);

    const existing = document.querySelector(".ml-settings-launcher");
    if (existing) {
      existing.remove();
    }
    const oldPanel = document.querySelector(".ml-settings-panel");
    if (oldPanel) {
      oldPanel.remove();
    }

    const launcher = document.createElement("button");
    launcher.type = "button";
    launcher.className = "ml-settings-launcher";
    launcher.setAttribute("aria-label", "開啟閱讀設定");
    launcher.textContent = "設定";

    let panel;
    const update = (patch) => {
      settings = { ...settings, ...patch };
      saveSettings(settings);
      applySettings(settings);
      panel.replaceWith(render());
      panel = document.querySelector(".ml-settings-panel");
      panel.hidden = false;
    };
    const render = () => renderPanel(settings, update);

    panel = render();
    launcher.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
    });

    document.body.append(launcher, panel);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(mount);
  }
})();
