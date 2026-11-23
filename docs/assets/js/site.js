(() => {
  const root = document.documentElement;
  root.classList.add("js");
  const themeButton = document.querySelector("[data-theme-toggle]");
  const navButton = document.querySelector("[data-nav-toggle]");
  const sidebar = document.querySelector(".sidebar");

  const savedTheme = localStorage.getItem("jimmytorch-docs-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    root.dataset.theme = savedTheme;
  }

  function renderThemeButton() {
    if (!themeButton) return;
    const dark = root.dataset.theme !== "light";
    themeButton.textContent = dark ? "☀" : "☾";
    themeButton.setAttribute("aria-label", dark ? "Use light theme" : "Use dark theme");
  }

  themeButton?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("jimmytorch-docs-theme", root.dataset.theme);
    renderThemeButton();
  });

  if (navButton && sidebar) {
    function setMenuOpen(open) {
      sidebar.classList.toggle("is-open", open);
      navButton.setAttribute("aria-expanded", String(open));
    }

    navButton.addEventListener("click", () => {
      const open = !sidebar.classList.contains("is-open");
      setMenuOpen(open);
      if (open) {
        sidebar.scrollTop = 0;
        sidebar.querySelector("a")?.focus({ preventScroll: true });
      }
    });

    function dismissOutside(event) {
      if (!sidebar.contains(event.target) && !navButton.contains(event.target)) setMenuOpen(false);
    }
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("focusin", dismissOutside);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && sidebar.classList.contains("is-open")) {
        setMenuOpen(false);
        navButton.focus({ preventScroll: true });
      }
    });
    window.matchMedia("(max-width: 900px)").addEventListener("change", () => setMenuOpen(false));
  }

  document.querySelectorAll("pre").forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code) return;
    const button = document.createElement("button");
    button.className = "copy-button";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent ?? "");
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select & copy";
      }
      window.setTimeout(() => { button.textContent = "Copy"; }, 1400);
    });
    pre.appendChild(button);
  });

  renderThemeButton();
})();
