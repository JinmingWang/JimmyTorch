const state = { snapshot: null, selectedMetric: null, polling: false };

const elements = {
  connection: document.querySelector("#connection-state"),
  progress: document.querySelector("#overall-progress"),
  percent: document.querySelector("#overall-percent"),
  progressBar: document.querySelector("#overall-progress-bar"),
  progressTrack: document.querySelector(".progress-track"),
  epoch: document.querySelector("#epoch-progress"),
  metrics: document.querySelector("#metric-cards"),
  metricSelect: document.querySelector("#metric-select"),
  chart: document.querySelector("#metric-chart"),
  chartEmpty: document.querySelector("#chart-empty"),
  eventList: document.querySelector("#event-list"),
  lrForm: document.querySelector("#learning-rate-form"),
  lrInput: document.querySelector("#learning-rate"),
  appliedLr: document.querySelector("#applied-learning-rate"),
  pendingLr: document.querySelector("#pending-learning-rate"),
  lrMessage: document.querySelector("#learning-rate-message"),
  theme: document.querySelector("#theme-select")
};

function formatNumber(value, precision = 4) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  if (Math.abs(value) !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 10000)) return value.toExponential(3);
  return value.toFixed(precision);
}

function formatScientific(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return value.toExponential(3);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function setConnection(text, className) {
  elements.connection.textContent = text;
  elements.connection.className = `connection-state ${className}`;
}

function render(snapshot) {
  state.snapshot = snapshot;
  const { progress, learning_rate: learningRate, custom_fields: fields } = snapshot;
  const percentage = Math.max(0, Math.min(100, progress.percent));
  elements.progress.textContent = `${progress.overall.toLocaleString()} / ${progress.total.toLocaleString()} batches`;
  elements.percent.textContent = `${percentage.toFixed(1)}%`;
  elements.progressBar.style.width = `${percentage}%`;
  elements.progressTrack.setAttribute("aria-valuenow", percentage.toFixed(1));
  elements.epoch.textContent = `Epoch ${progress.epoch} / ${progress.epochs}`;
  elements.appliedLr.textContent = formatNumber(learningRate.applied, 6);
  elements.pendingLr.textContent = learningRate.pending === null ? "None" : formatNumber(learningRate.pending, 6);
  renderMetricCards(progress, fields);
  updateMetricSelect(fields, chartEvents(snapshot));
  renderChart(chartEvents(snapshot));
  renderEvents(snapshot.recent_events);
}

function renderMetricCards(progress, fields) {
  const latest = state.snapshot.recent_events.at(-1)?.values || {};
  const cards = [
    ["Elapsed", formatDuration(progress.elapsed), "accent-cyan"],
    ["Remaining", formatDuration(progress.remaining), "accent-lime"],
    ["Throughput", `${formatNumber(progress.rate, 2)} batch/s`, "accent-yellow"],
    ...fields.map((field, index) => [field, formatNumber(latest[field]), ["accent-coral", "accent-cyan", "accent-lime", "accent-yellow"][index % 4]])
  ];
  elements.metrics.replaceChildren(...cards.map(([label, value, accent]) => {
    const card = document.createElement("article");
    card.className = `metric-card ${accent}`;
    const title = document.createElement("p");
    title.textContent = label;
    const number = document.createElement("strong");
    number.textContent = value;
    card.append(title, number);
    return card;
  }));
}

function chartEvents(snapshot) {
  return snapshot.recent_events;
}

function updateMetricSelect(fields, events) {
  const available = fields.filter(field => events.some(event => Number.isFinite(event.values[field])));
  if (!available.includes(state.selectedMetric)) state.selectedMetric = available[0] || null;
  const currentOptions = [...elements.metricSelect.options].map(option => option.value);
  if (currentOptions.join("|") !== available.join("|")) {
    elements.metricSelect.replaceChildren(...available.map(field => {
      const option = document.createElement("option");
      option.value = field;
      option.textContent = field;
      return option;
    }));
  }
  elements.metricSelect.value = state.selectedMetric || "";
  elements.metricSelect.disabled = available.length === 0;
}

function renderChart(events) {
  const canvas = elements.chart;
  const metric = state.selectedMetric;
  const points = metric ? events.filter(event => Number.isFinite(event.values[metric])) : [];
  elements.chartEmpty.style.display = points.length ? "none" : "block";
  canvas.style.display = points.length ? "block" : "none";
  if (!points.length) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const context = canvas.getContext("2d");
  context.scale(pixelRatio, pixelRatio);
  const styles = getComputedStyle(document.documentElement);
  const padding = { top: 16, right: 16, bottom: 28, left: 58 };
  const values = points.map(point => point.values[metric]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max) * 0.1, 1);
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = styles.getPropertyValue("--line");
  context.fillStyle = styles.getPropertyValue("--muted");
  context.font = "12px Cascadia Mono, monospace";
  for (let index = 0; index < 4; index += 1) {
    const y = padding.top + (plotHeight / 3) * index;
    const labelValue = max - ((span / 3) * index);
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
    context.fillText(formatScientific(labelValue), 2, y + 4);
  }
  context.strokeStyle = styles.getPropertyValue("--cyan");
  context.lineWidth = 2;
  context.beginPath();
  points.forEach((point, index) => {
    const x = padding.left + (points.length === 1 ? plotWidth / 2 : (plotWidth * index) / (points.length - 1));
    const y = padding.top + ((max - point.values[metric]) / span) * plotHeight;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.fillStyle = styles.getPropertyValue("--text");
  context.fillText(`Step ${points[0].global_step}`, padding.left, height - 8);
  context.fillText(`Step ${points.at(-1).global_step}`, Math.max(padding.left, width - padding.right - 92), height - 8);
}

function renderEvents(events) {
  const recent = events.slice(-6).reverse();
  elements.eventList.replaceChildren(...recent.map(event => {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = `Epoch ${event.epoch}, batch ${event.step}`;
    const details = document.createElement("span");
    const values = Object.entries(event.values).map(([name, value]) => `${name}: ${formatNumber(value)}`);
    details.textContent = `Global ${event.global_step} | ${values.join(" | ") || "No metrics"}`;
    item.append(heading, details);
    return item;
  }));
}

async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed (${response.status})`);
    const snapshot = await response.json();
    setConnection(snapshot.status === "closed" ? "Run completed" : "Live", "connected");
    render(snapshot);
    window.setTimeout(poll, Math.max(250, snapshot.refresh_interval * 1000));
  } catch (_error) {
    setConnection("Reconnecting", "error");
    window.setTimeout(poll, 1500);
  } finally {
    state.polling = false;
  }
}

elements.metricSelect.addEventListener("change", event => { state.selectedMetric = event.target.value; renderChart(state.snapshot ? chartEvents(state.snapshot) : []); });
window.addEventListener("resize", () => renderChart(state.snapshot ? chartEvents(state.snapshot) : []));
elements.lrForm.addEventListener("submit", async event => {
  event.preventDefault();
  const learningRate = Number(elements.lrInput.value);
  elements.lrMessage.className = "control-message";
  if (!window.confirm(`Apply learning rate ${learningRate} before the next training batch?`)) {
    elements.lrMessage.textContent = "Learning rate change cancelled.";
    return;
  }
  try {
    const response = await fetch("/api/learning-rate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ learning_rate: learningRate }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to queue learning rate.");
    elements.lrMessage.textContent = payload.message;
    elements.lrMessage.className = "control-message success";
  } catch (error) {
    elements.lrMessage.textContent = error.message;
    elements.lrMessage.className = "control-message error";
  }
});

function setTheme(theme) {
  document.documentElement.dataset.theme = theme === "system" ? "" : theme;
  localStorage.setItem("jimmytorch-progress-theme", theme);
  elements.theme.value = theme;
  renderChart(state.snapshot ? chartEvents(state.snapshot) : []);
}
elements.theme.addEventListener("change", event => setTheme(event.target.value));
setTheme(localStorage.getItem("jimmytorch-progress-theme") || "system");
poll();