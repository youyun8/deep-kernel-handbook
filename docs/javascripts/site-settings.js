(function () {
  const storageKey = "deep-kernel-handbook-settings";
  const defaults = {
    textSize: "standard", // small | standard | large
    width: "standard", // standard | wide
    theme: "auto", // auto | light | dark
    codeWrap: false,
  };

  const media =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-color-scheme: dark)")
      : null;

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

  function resolveScheme(theme) {
    if (theme === "light") return "default";
    if (theme === "dark") return "slate";
    return media && media.matches ? "slate" : "default";
  }

  function applySettings(settings) {
    const root = document.documentElement;
    root.dataset.readingText = settings.textSize;
    root.dataset.readingWidth = settings.width;
    root.dataset.codeWrap = String(Boolean(settings.codeWrap));

    // Drive the Material color scheme directly so the panel is the single
    // source of truth and "auto" can track prefers-color-scheme live.
    const scheme = resolveScheme(settings.theme);
    const body = document.body;
    if (body) {
      body.setAttribute("data-md-color-scheme", scheme);
      if (!body.getAttribute("data-md-color-primary")) {
        body.setAttribute("data-md-color-primary", "indigo");
        body.setAttribute("data-md-color-accent", "blue");
      }
    }
    // Keep Material's persisted palette in sync so its own toggle agrees.
    try {
      localStorage.setItem(
        "__palette",
        JSON.stringify({ index: scheme === "slate" ? 2 : 1, color: { scheme } })
      );
    } catch (_error) {
      /* ignore */
    }
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
    panel.setAttribute("aria-label", "外觀設定");
    panel.hidden = true;
    panel.innerHTML = `
      <div class="ml-settings-panel__head">
        <div>
          <p>外觀</p>
          <strong>外觀設定</strong>
        </div>
        <button type="button" class="ml-settings-close" aria-label="關閉設定">×</button>
      </div>
      <div class="ml-settings-row" data-setting="textSize">
        <span>文字大小</span>
        <div class="ml-settings-options ml-settings-options--three"></div>
      </div>
      <div class="ml-settings-row" data-setting="width">
        <span>內容寬度</span>
        <div class="ml-settings-options"></div>
      </div>
      <div class="ml-settings-row" data-setting="theme">
        <span>顏色主題</span>
        <div class="ml-settings-options ml-settings-options--three"></div>
      </div>
      <label class="ml-settings-check">
        <input type="checkbox" />
        <span>程式碼自動換行</span>
      </label>
    `;

    const textSize = panel.querySelector(
      '[data-setting="textSize"] .ml-settings-options'
    );
    textSize.append(
      button("小", settings.textSize === "small", () =>
        onChange({ textSize: "small" })
      ),
      button("標準", settings.textSize === "standard", () =>
        onChange({ textSize: "standard" })
      ),
      button("大", settings.textSize === "large", () =>
        onChange({ textSize: "large" })
      )
    );

    const width = panel.querySelector(
      '[data-setting="width"] .ml-settings-options'
    );
    width.append(
      button("標準", settings.width === "standard", () =>
        onChange({ width: "standard" })
      ),
      button("寬", settings.width === "wide", () =>
        onChange({ width: "wide" })
      )
    );

    const theme = panel.querySelector(
      '[data-setting="theme"] .ml-settings-options'
    );
    theme.append(
      button("自動", settings.theme === "auto", () =>
        onChange({ theme: "auto" })
      ),
      button("淺色", settings.theme === "light", () =>
        onChange({ theme: "light" })
      ),
      button("深色", settings.theme === "dark", () =>
        onChange({ theme: "dark" })
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
    launcher.setAttribute("aria-label", "開啟外觀設定");
    launcher.textContent = "外觀";

    let panel;
    const update = (patch) => {
      settings = { ...settings, ...patch };
      saveSettings(settings);
      applySettings(settings);
      const wasHidden = panel.hidden;
      const next = render();
      next.hidden = wasHidden;
      panel.replaceWith(next);
      panel = next;
    };
    const render = () => renderPanel(settings, update);

    panel = render();
    launcher.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
    });

    document.body.append(launcher, panel);
  }

  // Live-update the "auto" theme when the OS preference changes.
  if (media) {
    const onChange = () => {
      const settings = loadSettings();
      if (settings.theme === "auto") {
        applySettings(settings);
      }
    };
    if (media.addEventListener) {
      media.addEventListener("change", onChange);
    } else if (media.addListener) {
      media.addListener(onChange);
    }
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
