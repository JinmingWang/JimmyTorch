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

  navButton?.addEventListener("click", () => {
    const open = sidebar?.classList.toggle("is-open") ?? false;
    navButton.setAttribute("aria-expanded", String(open));
  });

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
