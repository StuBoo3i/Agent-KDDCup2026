const state = {
  runs: [],
  selectedRunId: null,
  selectedTaskId: null,
  config: null,
  filters: {
    status: "all",
    score: "all",
    labelKeyword: "",
  },
};
let filterDebounceTimer = null;

const byId = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function chip(value, trueLabel = "true", falseLabel = "false") {
  if (value === true) return `<span class="chip ok">${trueLabel}</span>`;
  if (value === false) return `<span class="chip fail">${falseLabel}</span>`;
  return `<span class="chip warn">N/A</span>`;
}

function statusChip(status) {
  if (status === "success") return `<span class="chip ok">success</span>`;
  if (status === "failed") return `<span class="chip fail">failed</span>`;
  return `<span class="chip warn">${esc(status || "unknown")}</span>`;
}

function reviewChip(review) {
  if (!review) return `<span class="chip neutral">未评价</span>`;
  const label = String(review.label || "").trim();
  const score = Number(review.score || 0);
  const scoreText = Number.isFinite(score) && score >= 1 && score <= 5 ? `${score}/5` : "未评分";
  if (label) {
    return `<span class="chip review">${esc(scoreText)} · ${esc(label)}</span>`;
  }
  return `<span class="chip review">${esc(scoreText)}</span>`;
}

function formatSec(value) {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return "-";
  return `${n.toFixed(3)}s`;
}

function roleClass(role) {
  const r = String(role || "").toLowerCase();
  if (r === "system") return "system";
  if (r === "assistant") return "assistant";
  if (r === "user") return "user";
  return "other";
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText} ${text}`);
  }
  return await res.json();
}

function getFilteredRuns() {
  const statusFilter = String(state.filters.status || "all");
  const scoreFilter = String(state.filters.score || "all");
  const labelKeyword = String(state.filters.labelKeyword || "").trim().toLowerCase();

  return state.runs.filter((run) => {
    if (statusFilter !== "all" && String(run.run_status || "") !== statusFilter) {
      return false;
    }

    const runScore = run.review?.score;
    if (scoreFilter === "unrated") {
      const parsed = Number(runScore);
      if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 5) {
        return false;
      }
    } else if (scoreFilter !== "all") {
      const expected = Number(scoreFilter);
      if (Number(runScore) !== expected) {
        return false;
      }
    }

    if (labelKeyword) {
      const label = String(run.review?.label || "").toLowerCase();
      if (!label.includes(labelKeyword)) {
        return false;
      }
    }

    return true;
  });
}

function renderRunCards() {
  const box = byId("runsList");
  const filteredRuns = getFilteredRuns();
  byId("runCountBadge").textContent = `${filteredRuns.length}/${state.runs.length}`;
  box.innerHTML = "";
  if (!filteredRuns.length) {
    box.innerHTML = state.runs.length
      ? `<p class="muted">当前筛选条件下没有匹配的 run。</p>`
      : `<p class="muted">没有找到 run 目录。</p>`;
    return;
  }

  for (const run of filteredRuns) {
    const div = document.createElement("div");
    div.className = "run-card" + (run.run_id === state.selectedRunId ? " active" : "");
    div.innerHTML = `
      <div class="run-id">${esc(run.run_id)}</div>
      <div class="run-meta">
        <div>状态: ${statusChip(run.run_status)}</div>
        <div>评价: ${reviewChip(run.review)}</div>
        <div>更新: ${esc(run.modified_at)}</div>
        <div>tasks: ${run.task_count}</div>
        <div>成功: ${run.task_succeeded_count}</div>
        <div>exact: ${run.eval_summary.exact_match_count ?? "-"}</div>
        <div>rate: ${run.eval_summary.exact_match_rate != null ? (run.eval_summary.exact_match_rate * 100).toFixed(1) + "%" : "-"}</div>
      </div>
    `;
    div.onclick = async () => {
      state.selectedRunId = run.run_id;
      state.selectedTaskId = null;
      renderRunCards();
      await loadRunDetail(run.run_id);
    };
    box.appendChild(div);
  }
}

function renderRunStats(run) {
  const stats = [
    { label: "Task 总数", value: run.task_count },
    { label: "运行成功", value: run.task_succeeded_count },
    { label: "Exact Match", value: run.eval_summary.exact_match_count ?? "-" },
    {
      label: "Exact Rate",
      value:
        run.eval_summary.exact_match_rate != null
          ? `${(run.eval_summary.exact_match_rate * 100).toFixed(1)}%`
          : "-",
    },
  ];
  byId("runStats").innerHTML = stats
    .map(
      (item) => `
      <div class="stat-item">
        <div class="label">${esc(item.label)}</div>
        <div class="value">${esc(item.value)}</div>
      </div>
    `
    )
    .join("");
}

function renderRunReview(run) {
  const review = run.review || {};
  const score = review.score == null ? "" : String(review.score);
  byId("reviewScore").value = score;
  byId("reviewLabel").value = review.label || "";
  byId("reviewNote").value = review.note || "";
  byId("reviewStatus").textContent = review.updated_at ? `最近更新: ${review.updated_at}` : "尚未评价";
}

function setReviewControlsEnabled(enabled) {
  byId("reviewScore").disabled = !enabled;
  byId("reviewLabel").disabled = !enabled;
  byId("reviewNote").disabled = !enabled;
  byId("saveReviewBtn").disabled = !enabled;
  byId("deleteRunBtn").disabled = !enabled;
}

function clearTaskDetail() {
  const tbody = byId("taskTable").querySelector("tbody");
  tbody.innerHTML = "";
  byId("taskTitle").textContent = "Trace 瀑布图";
  byId("taskMeta").textContent = "";
  byId("traceWaterfall").className = "waterfall-empty";
  byId("traceWaterfall").textContent = "请选择 task 查看 trace";
  byId("stepDetails").innerHTML = "";
  byId("predictionBox").innerHTML = `<p class="muted">暂无 prediction.csv</p>`;
  byId("goldBox").innerHTML = `<p class="muted">暂无 gold.csv</p>`;
  byId("mermaidBox").textContent = "暂无 graph.mmd";
}

function syncSelectionWithFilters() {
  const filteredRuns = getFilteredRuns();
  if (state.selectedRunId && !filteredRuns.some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = null;
    state.selectedTaskId = null;
  }
  if (!state.selectedRunId && filteredRuns.length) {
    state.selectedRunId = filteredRuns[0].run_id;
    state.selectedTaskId = null;
  }
}

function renderTaskTable(run) {
  const tbody = byId("taskTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const task of run.tasks) {
    const tr = document.createElement("tr");
    if (task.task_id === state.selectedTaskId) tr.classList.add("active");
    tr.innerHTML = `
      <td><code>${esc(task.task_id)}</code></td>
      <td>${chip(task.succeeded, "ok", "failed")}</td>
      <td>${chip(task.exact_match, "match", "mismatch")}</td>
      <td>${chip(task.columns_match, "ok", "bad")}</td>
      <td>${chip(task.unordered_row_match, "ok", "bad")}</td>
      <td>${task.step_count ?? 0}</td>
      <td>${esc(task.eval_error || "-")}</td>
    `;
    tr.onclick = async () => {
      state.selectedTaskId = task.task_id;
      renderTaskTable(run);
      await loadTaskDetail(run.run_id, task.task_id);
    };
    tbody.appendChild(tr);
  }
}

function renderWaterfall(task) {
  const box = byId("traceWaterfall");
  const meta = byId("taskMeta");
  meta.textContent =
    `succeeded=${task.succeeded} · ` +
    `e2e=${task.e2e_elapsed_seconds ?? "-"}s · ` +
    `steps=${task.step_count} · ` +
    `step_sum=${task.traced_step_seconds ?? "-"}s · ` +
    `overhead=${task.non_step_overhead_seconds ?? "-"}s`;

  if (!task.timeline || !task.timeline.length) {
    box.className = "waterfall-empty";
    box.textContent = "没有步骤数据";
    byId("stepDetails").innerHTML = "";
    return;
  }

  const total = Math.max(...task.timeline.map((step) => Number(step.end_s || 0)), 1);
  box.className = "waterfall";
  box.innerHTML = task.timeline
    .map((step) => {
      const left = (Number(step.start_s || 0) / total) * 100;
      const width = Math.max((Number(step.duration_s || 0) / total) * 100, 2.8);
      const cls = step.ok ? "wf-bar" : "wf-bar fail";
      return `
        <div class="wf-row">
          <div class="wf-label">#${esc(step.step_index)} · ${esc(step.action)}</div>
          <div class="wf-track"><div class="${cls}" style="left:${left}%;width:${width}%"></div></div>
          <div class="wf-time">${formatSec(step.duration_s)}</div>
        </div>
      `;
    })
    .join("");

  byId("stepDetails").innerHTML = task.timeline
    .map((step) => {
      const actionInput = esc(JSON.stringify(step.action_input, null, 2));
      const observation = esc(step.observation_preview || "");
      const rawResponse = esc(step.raw_response_preview || "");
      const thought = esc(step.thought || "(empty thought)");
      const promptMessages = Array.isArray(step.prompt_messages_preview) ? step.prompt_messages_preview : [];
      const renderPromptItems = (messages, indexMap, openCount = 1) =>
        messages
          .map((msg, localIdx) => {
            const role = String(msg.role || "unknown");
            const content = String(msg.content || "");
            const preview = esc(content.replace(/\s+/g, " ").slice(0, 90) || "(empty)");
            const globalIdx = indexMap[localIdx] ?? localIdx;
            const open = localIdx < openCount ? "open" : "";
            const codeStyle =
              content.length > 12000
                ? ' style="max-height:70vh;overflow:auto;"'
                : content.length > 5000
                  ? ' style="max-height:55vh;overflow:auto;"'
                  : "";
            return `
              <details class="prompt-msg" ${open}>
                <summary class="prompt-msg-summary">
                  <span class="prompt-msg-role role-${roleClass(role)}">${esc(role)}</span>
                  <span class="prompt-msg-meta">#${globalIdx + 1} · ${content.length} chars</span>
                  <span class="prompt-msg-preview">${preview}</span>
                </summary>
                <pre class="prompt-msg-code"${codeStyle}>${esc(content)}</pre>
              </details>
            `;
          })
          .join("");

      const totalPromptCount = step.prompt_message_count ?? promptMessages.length;
      const focusIndexSet = new Set();
      if (promptMessages.length > 0) {
        focusIndexSet.add(0); // system
      }
      if (promptMessages.length > 1) {
        focusIndexSet.add(1); // task user
      }
      if (promptMessages.length > 2) {
        focusIndexSet.add(promptMessages.length - 1); // latest
      }
      if (promptMessages.length > 3) {
        focusIndexSet.add(promptMessages.length - 2); // latest-1
      }
      const focusIndexes = Array.from(focusIndexSet).sort((a, b) => a - b);
      const focusMessages = focusIndexes.map((idx) => promptMessages[idx]).filter(Boolean);
      const focusRendered = focusMessages.length
        ? renderPromptItems(focusMessages, focusIndexes, 1)
        : `<p class="muted">该 step 未记录 prompt_messages（旧 run 或未开启）。</p>`;

      const fullRendered =
        promptMessages.length > 0
          ? renderPromptItems(
              promptMessages,
              promptMessages.map((_, idx) => idx),
              0
            )
          : "";

      const hiddenPromptCount = Math.max(0, promptMessages.length - focusMessages.length);
      const promptHeader = `Prompt Messages (${focusMessages.length}/${totalPromptCount})`;
      const fullPromptBlock =
        hiddenPromptCount > 0
          ? `
            <details class="prompt-more">
              <summary>查看全部 ${promptMessages.length} 条 Prompt Messages</summary>
              <div class="prompt-list prompt-list-full">${fullRendered}</div>
            </details>
          `
          : "";
      return `
      <div class="trace-card">
        <div class="trace-card-head">
          <div class="trace-step-title">Step #${esc(step.step_index)} · ${esc(step.action)}</div>
          <div class="trace-badges">
            ${step.ok ? '<span class="chip ok">ok</span>' : '<span class="chip fail">error</span>'}
            <span class="chip neutral">${formatSec(step.duration_s)}</span>
          </div>
        </div>
        <div class="trace-thought">${thought}</div>
        <div class="trace-grid">
          <div class="trace-block">
            <div class="trace-block-title">Action Input</div>
            <pre class="trace-code">${actionInput}</pre>
          </div>
          <div class="trace-block">
            <div class="trace-block-title">Observation</div>
            <pre class="trace-code">${observation}</pre>
          </div>
        </div>
        <div class="trace-block full">
          <div class="trace-block-title">${promptHeader}</div>
          <div class="prompt-list prompt-list-focus">${focusRendered}</div>
          ${fullPromptBlock}
        </div>
        <div class="trace-block full">
          <div class="trace-block-title">Raw Response</div>
          <pre class="trace-code">${rawResponse}</pre>
        </div>
      </div>
    `;
    })
    .join("");
}

function renderCsvPreview(boxId, data, emptyText, pathText) {
  const box = byId(boxId);
  const columns = data?.columns || [];
  const rows = data?.rows || [];
  if (!columns.length) {
    box.innerHTML = `<p class="muted">${esc(emptyText)}</p>${pathText ? `<p class="csv-path">${esc(pathText)}</p>` : ""}`;
    return;
  }
  const head = columns.map((c) => `<th>${esc(c)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`)
    .join("");
  box.innerHTML = `
    ${pathText ? `<p class="csv-path">${esc(pathText)}</p>` : ""}
    <table class="prediction-table">
      <thead><tr>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderPredictionAndGold(task) {
  renderCsvPreview(
    "predictionBox",
    task.prediction_preview,
    "暂无 prediction.csv",
    task.prediction_csv_path || ""
  );
  renderCsvPreview(
    "goldBox",
    task.gold_preview,
    "暂无 gold.csv",
    task.gold_csv_path || ""
  );
}

async function renderMermaid(mmd) {
  const box = byId("mermaidBox");
  if (!mmd || !mmd.trim()) {
    box.textContent = "暂无 graph.mmd";
    return;
  }
  try {
    const mermaid = window.__MERMAID__;
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      themeVariables: {
        primaryColor: "#d8f0f4",
        primaryTextColor: "#102027",
        primaryBorderColor: "#0b7285",
        lineColor: "#0b7285",
      },
    });
    const id = `mmd_${Date.now()}`;
    const { svg } = await mermaid.render(id, mmd);
    box.innerHTML = svg;
  } catch (err) {
    box.textContent = `Mermaid 渲染失败: ${err}`;
  }
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  syncSelectionWithFilters();
  renderRunCards();
  if (state.selectedRunId) {
    setReviewControlsEnabled(true);
    await loadRunDetail(state.selectedRunId);
    return;
  }
  setReviewControlsEnabled(false);
  byId("runTitle").textContent = "请选择一个 Run";
  byId("runMetaText").textContent = "";
  byId("runStats").innerHTML = "";
  renderRunReview({ review: {} });
  clearTaskDetail();
}

async function loadRunDetail(runId) {
  const payload = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
  const run = payload.run;
  state.selectedRunId = run.run_id;
  setReviewControlsEnabled(true);
  byId("runTitle").textContent = `Run: ${run.run_id}`;
  byId("runMetaText").textContent = `目录: ${run.run_dir}`;
  renderRunStats(run);
  renderRunReview(run);
  renderTaskTable(run);
  if (run.tasks.length) {
    const hasCurrentTask = !!state.selectedTaskId && run.tasks.some((item) => item.task_id === state.selectedTaskId);
    const taskId = hasCurrentTask ? state.selectedTaskId : run.tasks[0].task_id;
    state.selectedTaskId = taskId;
    renderTaskTable(run);
    await loadTaskDetail(runId, taskId);
    return;
  }
  clearTaskDetail();
}

async function loadTaskDetail(runId, taskId) {
  const payload = await fetchJson(`/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(taskId)}`);
  const task = payload.task;
  byId("taskTitle").textContent = `Trace 瀑布图: ${task.task_id}`;
  renderWaterfall(task);
  renderPredictionAndGold(task);
  await renderMermaid(task.graph_mmd);
}

async function cleanupFailedRuns() {
  const result = await fetchJson("/api/cleanup-failed", { method: "POST" });
  alert(`清理完成: deleted=${result.deleted_count}, skipped=${result.skipped.length}`);
  state.selectedRunId = null;
  state.selectedTaskId = null;
  await loadRuns();
}

async function saveRunReview() {
  if (!state.selectedRunId) {
    alert("请先选择一个 run");
    return;
  }
  const payload = {
    score: byId("reviewScore").value ? Number(byId("reviewScore").value) : null,
    label: byId("reviewLabel").value.trim(),
    note: byId("reviewNote").value.trim(),
  };
  const result = await fetchJson(`/api/runs/${encodeURIComponent(state.selectedRunId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  byId("reviewStatus").textContent = result.review?.updated_at
    ? `最近更新: ${result.review.updated_at}`
    : "评价已保存";
  await loadRuns();
}

async function deleteSelectedRun() {
  if (!state.selectedRunId) {
    alert("请先选择一个 run");
    return;
  }
  const runId = state.selectedRunId;
  const ok = window.confirm(`确认删除 run: ${runId} ? 这个操作不可恢复。`);
  if (!ok) {
    return;
  }
  await fetchJson(`/api/runs/${encodeURIComponent(runId)}/delete`, { method: "POST" });
  if (state.selectedRunId === runId) {
    state.selectedRunId = null;
    state.selectedTaskId = null;
  }
  await loadRuns();
}

async function onFiltersChanged() {
  state.filters.status = byId("filterStatus").value;
  state.filters.score = byId("filterScore").value;
  state.filters.labelKeyword = byId("filterLabel").value;
  syncSelectionWithFilters();
  renderRunCards();
  if (state.selectedRunId) {
    setReviewControlsEnabled(true);
    await loadRunDetail(state.selectedRunId);
    return;
  }
  setReviewControlsEnabled(false);
  byId("runTitle").textContent = "请选择一个 Run";
  byId("runMetaText").textContent = "";
  byId("runStats").innerHTML = "";
  renderRunReview({ review: {} });
  clearTaskDetail();
}

async function init() {
  setReviewControlsEnabled(false);
  byId("refreshBtn").onclick = async () => {
    try {
      await loadRuns();
    } catch (err) {
      alert(`刷新失败: ${err.message}`);
    }
  };
  byId("cleanupBtn").onclick = async () => {
    try {
      await cleanupFailedRuns();
    } catch (err) {
      alert(`清理失败: ${err.message}`);
    }
  };
  byId("saveReviewBtn").onclick = async () => {
    try {
      await saveRunReview();
    } catch (err) {
      alert(`保存评价失败: ${err.message}`);
    }
  };
  byId("deleteRunBtn").onclick = async () => {
    try {
      await deleteSelectedRun();
    } catch (err) {
      alert(`删除失败: ${err.message}`);
    }
  };
  byId("filterStatus").onchange = async () => {
    try {
      await onFiltersChanged();
    } catch (err) {
      alert(`筛选失败: ${err.message}`);
    }
  };
  byId("filterScore").onchange = async () => {
    try {
      await onFiltersChanged();
    } catch (err) {
      alert(`筛选失败: ${err.message}`);
    }
  };
  byId("filterLabel").oninput = () => {
    if (filterDebounceTimer) {
      clearTimeout(filterDebounceTimer);
    }
    filterDebounceTimer = setTimeout(async () => {
      try {
        await onFiltersChanged();
      } catch (err) {
        alert(`筛选失败: ${err.message}`);
      }
    }, 200);
  };

  state.config = await fetchJson("/api/config");
  if (state.config.auto_cleanup_on_page_load) {
    await cleanupFailedRuns();
  } else {
    await loadRuns();
  }
}

init().catch((err) => {
  console.error(err);
  alert(`初始化失败: ${err.message}`);
});
