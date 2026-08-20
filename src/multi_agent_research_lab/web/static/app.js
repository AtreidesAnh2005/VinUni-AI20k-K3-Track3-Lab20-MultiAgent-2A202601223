/**
 * Multi-Agent Research Lab - Interactive Web UI Controller
 */

// Pipeline steps definition
const PIPELINE_STEPS = ["supervisor", "researcher", "analyst", "writer", "critic"];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPresets();
  initCorpusSelect();
  initForm();
  loadBenchmarkReport();
});

async function initCorpusSelect() {
  const corpusSelect = document.getElementById("corpus-select");
  const queryInput = document.getElementById("query-input");
  if (!corpusSelect) return;

  try {
    const resp = await fetch("/api/corpus/topics");
    if (resp.ok) {
      const data = await resp.json();
      if (data.topics && data.topics.length > 0) {
        corpusSelect.innerHTML = `<option value="">-- Select from 30 benchmark corpus topics --</option>`;
        data.topics.forEach((t) => {
          const opt = document.createElement("option");
          opt.value = t.research_question || t.title;
          opt.textContent = `[Topic ${String(t.topic_number).padStart(2, "0")}] ${t.title}`;
          corpusSelect.appendChild(opt);
        });

        corpusSelect.addEventListener("change", () => {
          if (corpusSelect.value) {
            queryInput.value = corpusSelect.value;
            queryInput.focus();
          }
        });
      }
    }
  } catch (err) {
    console.debug("Failed to load corpus topics:", err);
  }
}

function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

function initPresets() {
  const chips = document.querySelectorAll(".preset-chip");
  const queryInput = document.getElementById("query-input");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-query") || chip.textContent.trim();
      queryInput.focus();
    });
  });
}

function initForm() {
  const runBtn = document.getElementById("run-btn");
  const queryInput = document.getElementById("query-input");

  runBtn.addEventListener("click", () => executeResearch());
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      executeResearch();
    }
  });
}

async function executeResearch() {
  const queryInput = document.getElementById("query-input");
  const runBtn = document.getElementById("run-btn");
  const modeSelect = document.getElementById("mode-select");
  const audienceSelect = document.getElementById("audience-select");
  const mockToggle = document.getElementById("mock-toggle");
  const sourcesSelect = document.getElementById("sources-select");

  const query = queryInput.value.trim();
  if (!query) {
    alert("Please enter a research query.");
    return;
  }

  // Set UI state to running
  runBtn.disabled = true;
  runBtn.innerHTML = `<span>Running Agents...</span>`;
  resetPipelineNodes();
  startPipelineAnimation();

  try {
    const payload = {
      query: query,
      mode: modeSelect.value,
      audience: audienceSelect.value,
      max_sources: parseInt(sourcesSelect.value, 10),
      mock: mockToggle.checked,
    };

    const resp = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await resp.json();
    renderResults(data);
    completePipelineAnimation(data.route_history);
  } catch (err) {
    alert(`Error executing research: ${err.message}`);
    resetPipelineNodes();
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = `<span>Run Research</span>`;
  }
}

function resetPipelineNodes() {
  PIPELINE_STEPS.forEach((step) => {
    const node = document.getElementById(`node-${step}`);
    if (node) {
      node.classList.remove("active", "completed");
    }
  });
}

let animationInterval = null;
function startPipelineAnimation() {
  clearInterval(animationInterval);
  let stepIdx = 0;
  animationInterval = setInterval(() => {
    PIPELINE_STEPS.forEach((step, idx) => {
      const node = document.getElementById(`node-${step}`);
      if (node) {
        if (idx === stepIdx % PIPELINE_STEPS.length) {
          node.classList.add("active");
          node.classList.remove("completed");
        } else if (idx < stepIdx % PIPELINE_STEPS.length) {
          node.classList.remove("active");
          node.classList.add("completed");
        }
      }
    });
    stepIdx++;
  }, 1200);
}

function completePipelineAnimation(routeHistory) {
  clearInterval(animationInterval);
  PIPELINE_STEPS.forEach((step) => {
    const node = document.getElementById(`node-${step}`);
    if (node) {
      node.classList.remove("active");
      node.classList.add("completed");
    }
  });
}

function renderResults(data) {
  // 1. Update Metrics
  document.getElementById("stat-latency").textContent = `${data.latency_seconds.toFixed(2)}s`;
  document.getElementById("stat-cost").textContent = `$${data.total_cost_usd.toFixed(5)}`;
  document.getElementById("stat-quality").textContent = `${data.quality_score.toFixed(1)}/10`;
  document.getElementById("stat-citations").textContent = `${(data.citation_coverage * 100).toFixed(0)}%`;

  // 2. Render Final Report (Markdown)
  const reportContainer = document.getElementById("report-content");
  reportContainer.innerHTML = parseMarkdown(data.final_answer || "No report content generated.");

  // 3. Render Sources
  const sourcesContainer = document.getElementById("sources-content");
  if (data.sources && data.sources.length > 0) {
    sourcesContainer.innerHTML = `
      <div class="sources-grid">
        ${data.sources
          .map(
            (s, idx) => `
          <div class="source-item-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <span class="citation-tag">[${idx + 1}]</span>
              <span style="font-size: 11px; color: var(--text-muted);">${s.url ? new URL(s.url).hostname : "Local Knowledge"}</span>
            </div>
            <div class="source-title">${escapeHtml(s.title)}</div>
            ${s.url ? `<a class="source-url" href="${s.url}" target="_blank" rel="noopener">${s.url}</a>` : ""}
            <div class="source-snippet">${escapeHtml(s.snippet)}</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;
  } else {
    sourcesContainer.innerHTML = `<div class="empty-state"><div class="empty-icon">📂</div><p>No external sources retrieved for this single-turn query.</p></div>`;
  }

  // 4. Render Analysis Notes
  const analysisContainer = document.getElementById("analysis-content");
  analysisContainer.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <div class="markdown-body">
        ${parseMarkdown(data.analysis_notes || data.research_notes || "No analysis notes available.")}
      </div>
    </div>
  `;

  // 5. Render Critic Review
  const criticContainer = document.getElementById("critic-content");
  criticContainer.innerHTML = `
    <div class="critic-card">
      <h3 style="font-family: 'Outfit'; color: #FFF;">Peer Review & Quality Audit</h3>
      <div class="gauge-row">
        <div class="gauge-item">
          <div style="display: flex; justify-content: space-between; font-size: 13px;">
            <span>Citation Grounding Coverage</span>
            <span style="color: var(--primary); font-weight: 600;">${(data.citation_coverage * 100).toFixed(0)}%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: ${data.citation_coverage * 100}%"></div>
          </div>
        </div>
        <div class="gauge-item">
          <div style="display: flex; justify-content: space-between; font-size: 13px;">
            <span>Overall Quality Score</span>
            <span style="color: var(--success); font-weight: 600;">${data.quality_score.toFixed(1)} / 10.0</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: ${data.quality_score * 10}%"></div>
          </div>
        </div>
      </div>
      <div style="margin-top: 10px; font-size: 13px; color: var(--text-secondary);">
        <strong>Supervisor Routing Sequence:</strong> ${data.route_history.join(" ➔ ")} (Iterations: ${data.iteration})
      </div>
    </div>
  `;
}

async function loadBenchmarkReport() {
  try {
    const resp = await fetch("/api/benchmark-report");
    if (resp.ok) {
      const data = await resp.json();
      const container = document.getElementById("benchmark-content");
      if (container && data.report) {
        container.innerHTML = `<div class="markdown-body">${parseMarkdown(data.report)}</div>`;
      }
    }
  } catch (err) {
    console.warn("Failed to load benchmark report:", err);
  }
}

// Lightweight Markdown to HTML parser
function parseMarkdown(md) {
  if (!md) return "";
  let html = escapeHtml(md);

  // Headers
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

  // Citations [1], [2]
  html = html.replace(/\[(\d+)\]/g, '<span class="citation-tag">[$1]</span>');

  // Lists
  html = html.replace(/^\- (.*$)/gim, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gim, "<ul>$1</ul>");

  // Line breaks
  html = html.replace(/\n\n/g, "<p></p>");
  html = html.replace(/\n/g, "<br/>");

  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
