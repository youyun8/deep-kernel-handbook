(function () {
  const storageKey = "deep-kernel-handbook-settings";
  const defaultSidebarWidth = 360;
  const sidebarLayoutVersion = 3;
  const defaults = {
    textSize: "standard", // small | standard | large
    width: "standard", // standard | wide
    theme: "auto", // auto | light | dark
    codeWrap: false,
    sidebarWidth: defaultSidebarWidth,
    sidebarCollapsed: false,
    sidebarLayoutVersion,
  };

  const media =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-color-scheme: dark)")
      : null;

  function sidebarWidthBounds() {
    const viewport = Math.max(document.documentElement.clientWidth, 0);
    return {
      min: 300,
      max: Math.min(560, Math.max(defaultSidebarWidth, viewport * 0.38)),
    };
  }

  function clampSidebarWidth(value) {
    const { min, max } = sidebarWidthBounds();
    const width = Number(value);
    if (!Number.isFinite(width)) return null;
    return Math.round(Math.min(Math.max(width, min), max));
  }

  function loadSettings() {
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || "{}");
      const settings = {
        ...defaults,
        ...stored,
      };
      if (stored.sidebarLayoutVersion !== sidebarLayoutVersion) {
        settings.sidebarWidth = defaultSidebarWidth;
        settings.sidebarLayoutVersion = sidebarLayoutVersion;
        localStorage.setItem(storageKey, JSON.stringify(settings));
      }
      return settings;
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
    root.dataset.sidebarCollapsed = String(Boolean(settings.sidebarCollapsed));
    const sidebarWidth = clampSidebarWidth(settings.sidebarWidth);
    if (sidebarWidth) {
      root.style.setProperty("--ml-sidebar-width", `${sidebarWidth}px`);
    } else {
      root.style.removeProperty("--ml-sidebar-width");
    }

    // Drive the Material color scheme directly so the panel is the single
    // source of truth and "auto" can track prefers-color-scheme live. The
    // scheme lives on <html> (set pre-paint by overrides/main.html and never
    // replaced by instant navigation) with <body> kept in sync for parity.
    const scheme = resolveScheme(settings.theme);
    root.setAttribute("data-md-color-scheme", scheme);
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
        JSON.stringify({
          index: scheme === "slate" ? 2 : 1,
          color: { scheme },
        }),
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

  function renderPanel(settings, onChange, onClose) {
    const overlay = document.createElement("div");
    overlay.className = "ml-settings-overlay";
    overlay.hidden = true;
    overlay.innerHTML = `
      <aside class="ml-settings-panel" id="ml-settings-panel" role="dialog" aria-modal="true" aria-label="外觀設定">
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
      </aside>
    `;

    const textSize = overlay.querySelector(
      '[data-setting="textSize"] .ml-settings-options',
    );
    textSize.append(
      button("小", settings.textSize === "small", () =>
        onChange({ textSize: "small" }),
      ),
      button("標準", settings.textSize === "standard", () =>
        onChange({ textSize: "standard" }),
      ),
      button("大", settings.textSize === "large", () =>
        onChange({ textSize: "large" }),
      ),
    );

    const width = overlay.querySelector(
      '[data-setting="width"] .ml-settings-options',
    );
    width.append(
      button("標準", settings.width === "standard", () =>
        onChange({ width: "standard" }),
      ),
      button("寬", settings.width === "wide", () =>
        onChange({ width: "wide" }),
      ),
    );

    const theme = overlay.querySelector(
      '[data-setting="theme"] .ml-settings-options',
    );
    theme.append(
      button("自動", settings.theme === "auto", () =>
        onChange({ theme: "auto" }),
      ),
      button("淺色", settings.theme === "light", () =>
        onChange({ theme: "light" }),
      ),
      button("深色", settings.theme === "dark", () =>
        onChange({ theme: "dark" }),
      ),
    );

    const check = overlay.querySelector("input");
    check.checked = Boolean(settings.codeWrap);
    check.addEventListener("change", () =>
      onChange({ codeWrap: check.checked }),
    );
    overlay
      .querySelector(".ml-settings-close")
      .addEventListener("click", onClose);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        onClose();
      }
    });
    return overlay;
  }

  function mountSidebarResize(settings, onChange, controller) {
    const oldHandle = document.querySelector(".ml-sidebar-resize");
    if (oldHandle) {
      oldHandle.remove();
    }

    const sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;

    const handle = document.createElement("div");
    handle.className = "ml-sidebar-resize";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-label", "Resize sidebar");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("tabindex", "0");
    handle.title = "Resize sidebar";
    sidebar.append(handle);

    const setHandleValue = (width) => {
      const { min, max } = sidebarWidthBounds();
      const current =
        clampSidebarWidth(width) ||
        Math.round(sidebar.getBoundingClientRect().width);
      handle.setAttribute("aria-valuemin", String(Math.round(min)));
      handle.setAttribute("aria-valuemax", String(Math.round(max)));
      handle.setAttribute("aria-valuenow", String(current));
    };

    const applyWidth = (width) => {
      const next = clampSidebarWidth(width);
      if (!next) return null;
      document.documentElement.style.setProperty(
        "--ml-sidebar-width",
        `${next}px`,
      );
      setHandleValue(next);
      return next;
    };

    setHandleValue(settings.sidebarWidth);

    let drag = null;
    let storedSidebarWidth =
      clampSidebarWidth(settings.sidebarWidth) ||
      Math.round(sidebar.getBoundingClientRect().width);
    const startDrag = (clientX, pointerId = null) => {
      const current = Math.round(sidebar.getBoundingClientRect().width);
      drag = {
        pointerId,
        startX: clientX,
        startWidth: current,
        width: current,
      };
      document.body.classList.add("ml-sidebar-resizing");
    };
    const moveDrag = (clientX) => {
      if (!drag) return;
      drag.width = applyWidth(drag.startWidth + clientX - drag.startX);
    };
    const finishDrag = () => {
      if (!drag) return;
      const width = drag.width;
      drag = null;
      document.body.classList.remove("ml-sidebar-resizing");
      if (width) {
        storedSidebarWidth = width;
        onChange({ sidebarWidth: width });
      }
    };

    handle.addEventListener(
      "pointerdown",
      (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        startDrag(event.clientX, event.pointerId);
        handle.setPointerCapture(event.pointerId);
      },
      { signal: controller.signal },
    );

    handle.addEventListener(
      "pointermove",
      (event) => {
        if (!drag || drag.pointerId !== event.pointerId) return;
        moveDrag(event.clientX);
      },
      { signal: controller.signal },
    );

    const finishPointerDrag = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      finishDrag();
    };

    handle.addEventListener("pointerup", finishPointerDrag, {
      signal: controller.signal,
    });
    handle.addEventListener("pointercancel", finishPointerDrag, {
      signal: controller.signal,
    });

    handle.addEventListener(
      "mousedown",
      (event) => {
        if (event.button !== 0 || drag) return;
        event.preventDefault();
        startDrag(event.clientX);
      },
      { signal: controller.signal },
    );
    document.addEventListener(
      "mousemove",
      (event) => {
        if (!drag || drag.pointerId !== null) return;
        moveDrag(event.clientX);
      },
      { signal: controller.signal },
    );
    document.addEventListener(
      "mouseup",
      () => {
        if (!drag || drag.pointerId !== null) return;
        finishDrag();
      },
      { signal: controller.signal },
    );

    handle.addEventListener(
      "keydown",
      (event) => {
        const current = Math.round(sidebar.getBoundingClientRect().width);
        let next = null;
        if (event.key === "ArrowLeft") next = current - 16;
        if (event.key === "ArrowRight") next = current + 16;
        if (event.key === "Home") next = sidebarWidthBounds().min;
        if (event.key === "End") next = sidebarWidthBounds().max;
        if (next === null) return;
        event.preventDefault();
        const width = applyWidth(next);
        if (width) {
          onChange({ sidebarWidth: width });
        }
      },
      { signal: controller.signal },
    );

    window.addEventListener(
      "resize",
      () => {
        applySettings(loadSettings());
        setHandleValue(loadSettings().sidebarWidth);
      },
      { signal: controller.signal },
    );

    if (typeof ResizeObserver === "function") {
      const resizeObserver = new ResizeObserver((entries) => {
        if (drag) return;
        if (document.documentElement.dataset.sidebarCollapsed === "true") return;
        if (document.body.classList.contains("ml-sidebar-toggle-transition")) {
          return;
        }
        const entry = entries[0];
        if (!entry) return;
        const width = clampSidebarWidth(
          Math.round(sidebar.getBoundingClientRect().width),
        );
        if (!width || Math.abs(width - storedSidebarWidth) < 3) return;
        storedSidebarWidth = width;
        document.documentElement.style.setProperty(
          "--ml-sidebar-width",
          `${width}px`,
        );
        setHandleValue(width);
        onChange({ sidebarWidth: width });
      });
      resizeObserver.observe(sidebar);
      controller.signal.addEventListener("abort", () =>
        resizeObserver.disconnect(),
      );
    }
  }

  function syncSidebarToggle(collapsed) {
    document.querySelectorAll(".ml-sidebar-toggle").forEach((button) => {
      button.setAttribute("aria-pressed", String(Boolean(collapsed)));
      button.setAttribute(
        "aria-label",
        collapsed ? "展開左側導覽" : "收合左側導覽",
      );
      button.title = collapsed ? "展開左側導覽" : "收合左側導覽";
    });
  }

  function mountSidebarToggle(settings, onChange) {
    document
      .querySelectorAll(
        ".ml-sidebar-rail, .md-sidebar--primary .ml-sidebar-toggle",
      )
      .forEach((node) => node.remove());

    const sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return null;

    const toggle = () => {
      const collapsed =
        document.documentElement.dataset.sidebarCollapsed !== "true";
      document.body.classList.add("ml-sidebar-toggle-transition");
      window.setTimeout(
        () => document.body.classList.remove("ml-sidebar-toggle-transition"),
        360,
      );
      onChange({ sidebarCollapsed: collapsed });
    };

    const createButton = (variant) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `ml-sidebar-toggle ml-sidebar-toggle--${variant}`;
      button.innerHTML = `
        <svg class="ml-sidebar-toggle__open" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="4" width="18" height="16" rx="2"></rect>
          <path d="M9 4v16"></path>
          <path d="m14 9 3 3-3 3"></path>
        </svg>
        <svg class="ml-sidebar-toggle__close" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="4" width="18" height="16" rx="2"></rect>
          <path d="M9 4v16"></path>
          <path d="m17 9-3 3 3 3"></path>
        </svg>
      `;
      button.addEventListener("click", toggle);
      return button;
    };

    const navTitle = sidebar.querySelector(".md-nav--primary > .md-nav__title");
    if (navTitle) {
      navTitle.append(createButton("brand"));
    }

    const rail = document.createElement("div");
    rail.className = "ml-sidebar-rail";
    rail.append(createButton("expand"));
    document.body.append(rail);

    syncSidebarToggle(settings.sidebarCollapsed);
    return navTitle;
  }

  function mount() {
    let settings = loadSettings();
    let isOpen = false;
    applySettings(settings);

    const existing = document.querySelector(".ml-settings-button");
    if (existing) {
      existing.remove();
    }
    document
      .querySelectorAll(".ml-sidebar-rail, .ml-sidebar-toggle")
      .forEach((node) => node.remove());
    const oldOverlay = document.querySelector(".ml-settings-overlay");
    if (oldOverlay) {
      oldOverlay.remove();
    }
    if (window.__deepKernelSettingsAbort) {
      window.__deepKernelSettingsAbort.abort();
    }

    const launcher = document.createElement("button");
    launcher.type = "button";
    launcher.className = "ml-settings-button";
    launcher.setAttribute("aria-label", "開啟外觀設定");
    launcher.setAttribute("aria-controls", "ml-settings-panel");
    launcher.setAttribute("aria-expanded", "false");
    launcher.title = "外觀設定";
    launcher.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"></path>
        <path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.04.04a2 2 0 0 1-2.83 2.83l-.04-.04a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.1 1.65V21a2 2 0 0 1-4 0v-.06A1.8 1.8 0 0 0 8.75 19.3a1.8 1.8 0 0 0-1.98.36l-.04.04a2 2 0 0 1-2.83-2.83l.04-.04a1.8 1.8 0 0 0 .36-1.98 1.8 1.8 0 0 0-1.65-1.1H2.6a2 2 0 0 1 0-4h.06A1.8 1.8 0 0 0 4.3 8.65a1.8 1.8 0 0 0-.36-1.98l-.04-.04A2 2 0 0 1 6.73 3.8l.04.04a1.8 1.8 0 0 0 1.98.36 1.8 1.8 0 0 0 1.1-1.65V2.5a2 2 0 0 1 4 0v.06a1.8 1.8 0 0 0 1.1 1.65 1.8 1.8 0 0 0 1.98-.36l.04-.04a2 2 0 0 1 2.83 2.83l-.04.04a1.8 1.8 0 0 0-.36 1.98 1.8 1.8 0 0 0 1.65 1.1h.06a2 2 0 0 1 0 4h-.06A1.8 1.8 0 0 0 19.4 15Z"></path>
      </svg>
    `;

    let overlay;
    const close = () => {
      isOpen = false;
      overlay.hidden = true;
      document.body.classList.remove("ml-settings-open");
      launcher.setAttribute("aria-expanded", "false");
    };
    const open = () => {
      isOpen = true;
      overlay.hidden = false;
      document.body.classList.add("ml-settings-open");
      launcher.setAttribute("aria-expanded", "true");
      const closeButton = overlay.querySelector(".ml-settings-close");
      if (closeButton) closeButton.focus({ preventScroll: true });
    };
    const update = (patch) => {
      settings = { ...settings, ...patch };
      saveSettings(settings);
      applySettings(settings);
      syncSidebarToggle(settings.sidebarCollapsed);
      const wasHidden = overlay.hidden;
      const next = render();
      next.hidden = wasHidden;
      overlay.replaceWith(next);
      overlay = next;
    };
    const render = () => renderPanel(settings, update, close);

    overlay = render();
    mountSidebarToggle(settings, update);
    launcher.addEventListener("click", () => {
      if (isOpen) {
        close();
      } else {
        open();
      }
    });

    const header = document.querySelector(".md-header__inner");
    if (header) {
      header.insertBefore(
        launcher,
        header.querySelector('label[for="__search"]') ||
          header.querySelector(".md-search") ||
          header.querySelector(".md-header__source"),
      );
    } else {
      document.body.append(launcher);
    }
    document.body.append(overlay);

    const controller = new AbortController();
    window.__deepKernelSettingsAbort = controller;
    mountSidebarResize(settings, update, controller);
    document.addEventListener(
      "keydown",
      (event) => {
        if (event.key === "Escape" && isOpen) {
          close();
          launcher.focus({ preventScroll: true });
        }
      },
      { signal: controller.signal },
    );
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
