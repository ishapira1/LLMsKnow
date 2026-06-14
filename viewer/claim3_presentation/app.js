const BUNDLE_PATH = "./data/claim3_presentation_bundle_main_runs.json";

const state = {
  model: "All",
  dataset: "All",
  aggregationMode: "equal_weight",
  visibleMetricIds: new Set(),
  visibleProbeTrainOn: new Set(),
  visibleEvalOn: new Set(),
  sortMetricId: null,
  sortFieldId: null,
  sortDirection: "desc",
};

let bundle = null;

const elements = {
  modelSelect: document.getElementById("model-select"),
  datasetSelect: document.getElementById("dataset-select"),
  aggregationControl: document.getElementById("aggregation-control"),
  metricsPanel: document.getElementById("metrics-options"),
  trainOnPanel: document.getElementById("train-on-options"),
  evalOnPanel: document.getElementById("eval-on-options"),
  metricsSummary: document.getElementById("metrics-summary"),
  trainOnSummary: document.getElementById("train-on-summary"),
  evalOnSummary: document.getElementById("eval-on-summary"),
  statusLine: document.getElementById("status-line"),
  resetSortButton: document.getElementById("reset-sort-button"),
  resultsTable: document.getElementById("results-table"),
};

const titleize = (value) =>
  String(value || "")
    .split("_")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");

const buildViewKey = (model, dataset, aggregationMode) =>
  `model=${model}|dataset=${dataset}|aggregation=${aggregationMode}`;

const metricSpecById = (metricId) =>
  bundle.metric_specs.find((metric) => metric.id === metricId);

const rowMetricValue = (row, metricId) => {
  const metricSpec = metricSpecById(metricId);
  if (!metricSpec) {
    return null;
  }
  return row[metricSpec.family].metrics[metricSpec.metric_key];
};

const rowMetricAvailable = (row, metricId) => {
  const metricSpec = metricSpecById(metricId);
  if (!metricSpec) {
    return false;
  }
  return Boolean(row[metricSpec.family].available);
};

const formatMetric = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return Number(value).toFixed(3);
};

const formatCoverage = (payload) => `${payload.runs_contributing}/${payload.runs_selected}`;

const metricHeatColor = (metricId, intensity) => {
  if (intensity === null) {
    return "transparent";
  }
  const color = metricId.startsWith("probe_") ? [212, 101, 26] : [115, 179, 171];
  const alpha = 0.1 + intensity * 0.32;
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha.toFixed(3)})`;
};

const semanticRowCompare = (left, right) => left.semantic_order - right.semantic_order;

const semanticValueCompare = (fieldId, leftValue, rightValue) => {
  const order = bundle.semantic_orders[fieldId] || [];
  const leftIndex = order.indexOf(leftValue);
  const rightIndex = order.indexOf(rightValue);
  if (leftIndex !== rightIndex) {
    return leftIndex - rightIndex;
  }
  return String(leftValue || "").localeCompare(String(rightValue || ""));
};

const activeSortId = () => state.sortFieldId || state.sortMetricId;

const activeSortIndicator = (sortId) => {
  if (activeSortId() !== sortId) {
    return "↕";
  }
  return state.sortDirection === "desc" ? "↓" : "↑";
};

const activateMetricSort = (metricId) => {
  state.sortFieldId = null;
  if (state.sortMetricId === metricId) {
    state.sortDirection = state.sortDirection === "desc" ? "asc" : "desc";
  } else {
    state.sortMetricId = metricId;
    state.sortDirection = "desc";
  }
};

const activateFieldSort = (fieldId) => {
  state.sortMetricId = null;
  if (state.sortFieldId === fieldId) {
    state.sortDirection = state.sortDirection === "desc" ? "asc" : "desc";
  } else {
    state.sortFieldId = fieldId;
    state.sortDirection = "desc";
  }
};

const currentView = () => bundle.views[buildViewKey(state.model, state.dataset, state.aggregationMode)];

const currentVisibleMetrics = () =>
  bundle.metric_specs.filter((metric) => state.visibleMetricIds.has(metric.id));

const currentRows = () => {
  const rows = currentView().rows.filter(
    (row) =>
      state.visibleProbeTrainOn.has(row.probe_train_on) && state.visibleEvalOn.has(row.eval_on),
  );
  if (!state.sortMetricId && !state.sortFieldId) {
    return rows.slice().sort(semanticRowCompare);
  }

  const sorted = rows.slice().sort((left, right) => {
    if (state.sortFieldId) {
      const fieldCompare = semanticValueCompare(
        state.sortFieldId,
        left[state.sortFieldId],
        right[state.sortFieldId],
      );
      if (fieldCompare !== 0) {
        return state.sortDirection === "desc" ? -fieldCompare : fieldCompare;
      }
      return semanticRowCompare(left, right);
    }

    const leftAvailable = rowMetricAvailable(left, state.sortMetricId);
    const rightAvailable = rowMetricAvailable(right, state.sortMetricId);
    if (leftAvailable !== rightAvailable) {
      return leftAvailable ? -1 : 1;
    }
    if (!leftAvailable && !rightAvailable) {
      return semanticRowCompare(left, right);
    }

    const leftValue = rowMetricValue(left, state.sortMetricId);
    const rightValue = rowMetricValue(right, state.sortMetricId);
    if (leftValue !== rightValue) {
      return state.sortDirection === "desc" ? rightValue - leftValue : leftValue - rightValue;
    }
    return semanticRowCompare(left, right);
  });

  return sorted;
};

const metricRanges = (rows, visibleMetrics) => {
  const ranges = {};
  visibleMetrics.forEach((metric) => {
    const values = rows
      .map((row) => rowMetricValue(row, metric.id))
      .filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
    if (!values.length) {
      ranges[metric.id] = null;
      return;
    }
    ranges[metric.id] = {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  });
  return ranges;
};

const heatIntensity = (ranges, metricId, value) => {
  const range = ranges[metricId];
  if (!range || value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }
  if (range.max === range.min) {
    return 0.6;
  }
  return (value - range.min) / (range.max - range.min);
};

const updateSummary = (element, selectedCount, totalCount) => {
  element.textContent = `${selectedCount}/${totalCount}`;
};

const createSelectOptions = (selectElement, options, selectedValue) => {
  selectElement.innerHTML = "";
  options.forEach((option) => {
    const htmlOption = document.createElement("option");
    htmlOption.value = option;
    htmlOption.textContent = option;
    htmlOption.selected = option === selectedValue;
    selectElement.appendChild(htmlOption);
  });
};

const createSegmentedControl = () => {
  elements.aggregationControl.innerHTML = "";
  bundle.selector_options.aggregation_modes.forEach((mode) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `segmented-option${state.aggregationMode === mode ? " active" : ""}`;
    button.textContent = mode === "equal_weight" ? "Equal weight" : "Prompt weighted";
    button.addEventListener("click", () => {
      state.aggregationMode = mode;
      createSegmentedControl();
      render();
    });
    elements.aggregationControl.appendChild(button);
  });
};

const renderChecklist = ({ root, groups, selectedSet, onToggle }) => {
  root.innerHTML = "";

  const quickActions = document.createElement("div");
  quickActions.className = "quick-actions";
  const allButton = document.createElement("button");
  allButton.type = "button";
  allButton.textContent = "All";
  allButton.addEventListener("click", () => {
    groups.flatMap((group) => group.options).forEach((option) => selectedSet.add(option));
    onToggle();
  });
  const noneButton = document.createElement("button");
  noneButton.type = "button";
  noneButton.textContent = "None";
  noneButton.addEventListener("click", () => {
    selectedSet.clear();
    onToggle();
  });
  quickActions.append(allButton, noneButton);
  root.appendChild(quickActions);

  groups.forEach((group) => {
    const wrapper = document.createElement("div");
    wrapper.className = "checklist-group";
    if (group.label) {
      const heading = document.createElement("h3");
      heading.textContent = group.label;
      wrapper.appendChild(heading);
    }
    group.options.forEach((option) => {
      const label = document.createElement("label");
      label.className = "check-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selectedSet.has(option);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          selectedSet.add(option);
        } else {
          selectedSet.delete(option);
        }
        onToggle();
      });
      const text = document.createElement("span");
      text.textContent = group.labelMap ? group.labelMap[option] : titleize(option);
      label.append(checkbox, text);
      wrapper.appendChild(label);
    });
    root.appendChild(wrapper);
  });
};

const renderControls = () => {
  createSelectOptions(elements.modelSelect, bundle.selector_options.models, state.model);
  createSelectOptions(elements.datasetSelect, bundle.selector_options.datasets, state.dataset);
  createSegmentedControl();

  renderChecklist({
    root: elements.metricsPanel,
    selectedSet: state.visibleMetricIds,
    groups: bundle.metric_groups.map((group) => ({
      label: group.label,
      options: group.metric_ids,
      labelMap: Object.fromEntries(
        group.metric_ids.map((metricId) => {
          const metric = metricSpecById(metricId);
          return [metricId, metric.short_label];
        }),
      ),
    })),
    onToggle: () => {
      renderControls();
      updateSummary(
        elements.metricsSummary,
        state.visibleMetricIds.size,
        bundle.selector_options.metrics.length,
      );
      render();
    },
  });

  renderChecklist({
    root: elements.trainOnPanel,
    selectedSet: state.visibleProbeTrainOn,
    groups: [
      {
        label: "",
        options: bundle.selector_options.probe_train_on,
      },
    ],
    onToggle: () => {
      renderControls();
      updateSummary(
        elements.trainOnSummary,
        state.visibleProbeTrainOn.size,
        bundle.selector_options.probe_train_on.length,
      );
      render();
    },
  });

  renderChecklist({
    root: elements.evalOnPanel,
    selectedSet: state.visibleEvalOn,
    groups: [
      {
        label: "",
        options: bundle.selector_options.eval_on,
      },
    ],
    onToggle: () => {
      renderControls();
      updateSummary(
        elements.evalOnSummary,
        state.visibleEvalOn.size,
        bundle.selector_options.eval_on.length,
      );
      render();
    },
  });

  updateSummary(elements.metricsSummary, state.visibleMetricIds.size, bundle.selector_options.metrics.length);
  updateSummary(
    elements.trainOnSummary,
    state.visibleProbeTrainOn.size,
    bundle.selector_options.probe_train_on.length,
  );
  updateSummary(elements.evalOnSummary, state.visibleEvalOn.size, bundle.selector_options.eval_on.length);
};

const renderTable = () => {
  const rows = currentRows();
  const visibleMetrics = currentVisibleMetrics();
  const ranges = metricRanges(rows, visibleMetrics);
  const modelMetrics = visibleMetrics.filter((metric) => metric.family === "model");
  const probeMetrics = visibleMetrics.filter((metric) => metric.family === "probe");

  const thead = elements.resultsTable.querySelector("thead");
  const tbody = elements.resultsTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const groupRow = document.createElement("tr");
  const stickyBlank = document.createElement("th");
  stickyBlank.className = "group-header sticky-col sticky-col-first";
  stickyBlank.colSpan = 2;
  stickyBlank.textContent = "Rows";
  groupRow.appendChild(stickyBlank);

  if (modelMetrics.length) {
    const modelHeader = document.createElement("th");
    modelHeader.className = "group-header";
    modelHeader.colSpan = modelMetrics.length;
    modelHeader.textContent = "Model performance";
    groupRow.appendChild(modelHeader);
  }

  if (probeMetrics.length) {
    const probeHeader = document.createElement("th");
    probeHeader.className = "group-header";
    probeHeader.colSpan = probeMetrics.length;
    probeHeader.textContent = "Chosen probe performance";
    groupRow.appendChild(probeHeader);
  }

  thead.appendChild(groupRow);

  const columnRow = document.createElement("tr");
  const trainOnHeader = document.createElement("th");
  trainOnHeader.className = "sticky-col sticky-col-first";
  const trainOnButton = document.createElement("button");
  trainOnButton.type = "button";
  trainOnButton.className = `metric-header-button${state.sortFieldId === "probe_train_on" ? " active-sort" : ""}`;
  trainOnButton.innerHTML = `<span>Probe train on</span><span class="sort-indicator">${activeSortIndicator(
    "probe_train_on",
  )}</span>`;
  trainOnButton.addEventListener("click", () => {
    activateFieldSort("probe_train_on");
    render();
  });
  trainOnHeader.appendChild(trainOnButton);
  columnRow.appendChild(trainOnHeader);

  const evalOnHeader = document.createElement("th");
  evalOnHeader.className = "sticky-col sticky-col-second";
  const evalOnButton = document.createElement("button");
  evalOnButton.type = "button";
  evalOnButton.className = `metric-header-button${state.sortFieldId === "eval_on" ? " active-sort" : ""}`;
  evalOnButton.innerHTML = `<span>Eval on</span><span class="sort-indicator">${activeSortIndicator(
    "eval_on",
  )}</span>`;
  evalOnButton.addEventListener("click", () => {
    activateFieldSort("eval_on");
    render();
  });
  evalOnHeader.appendChild(evalOnButton);
  columnRow.appendChild(evalOnHeader);

  visibleMetrics.forEach((metric) => {
    const th = document.createElement("th");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `metric-header-button${state.sortMetricId === metric.id ? " active-sort" : ""}`;
    button.innerHTML = `<span>${metric.short_label}</span><span class="sort-indicator">${activeSortIndicator(
      metric.id,
    )}</span>`;
    button.addEventListener("click", () => {
      activateMetricSort(metric.id);
      render();
    });
    th.appendChild(button);
    columnRow.appendChild(th);
  });
  thead.appendChild(columnRow);

  if (!rows.length) {
    const emptyRow = document.createElement("tr");
    const emptyCell = document.createElement("td");
    emptyCell.colSpan = 2 + visibleMetrics.length;
    emptyCell.className = "cell-subtle";
    emptyCell.textContent = "No rows match the current filters.";
    emptyRow.appendChild(emptyCell);
    tbody.appendChild(emptyRow);

    const current = currentView();
    elements.statusLine.textContent = `${current.selected_run_count} runs selected, 0 row combinations visible, ${visibleMetrics.length} metrics shown.`;
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");

    const trainOnCell = document.createElement("td");
    trainOnCell.className = "sticky-col sticky-col-first";
    trainOnCell.innerHTML = `<div class="probe-train-label">${titleize(row.probe_train_on)}</div>`;
    tr.appendChild(trainOnCell);

    const evalOnCell = document.createElement("td");
    evalOnCell.className = "sticky-col sticky-col-second";
    evalOnCell.innerHTML = `
      <div class="eval-label">${titleize(row.eval_on)}</div>
      <div class="coverage-badges">
        <span class="coverage-badge">model ${formatCoverage(row.model)}</span>
        <span class="coverage-badge">probe ${formatCoverage(row.probe)}</span>
      </div>
    `;
    tr.appendChild(evalOnCell);

    visibleMetrics.forEach((metric) => {
      const cell = document.createElement("td");
      const available = rowMetricAvailable(row, metric.id);
      const value = rowMetricValue(row, metric.id);
      const familyPayload = row[metric.family];
      const tooltipLines = [
        `${metric.label}`,
        `Value: ${formatMetric(value)}`,
        `Coverage: ${familyPayload.runs_contributing}/${familyPayload.runs_selected} runs`,
        `Prompt weight: ${Number(familyPayload.prompt_weight_total || 0).toFixed(1)}`,
      ];
      cell.title = tooltipLines.join("\n");
      cell.className = `metric-cell${available ? "" : " metric-na"}`;
      if (available) {
        const intensity = heatIntensity(ranges, metric.id, value);
        cell.style.background = metricHeatColor(metric.id, intensity);
        cell.innerHTML = `
          <span class="metric-value">${formatMetric(value)}</span>
          <span class="metric-coverage">${formatCoverage(familyPayload)}</span>
        `;
      } else {
        cell.innerHTML = `<span class="metric-value">N/A</span><span class="metric-coverage">${formatCoverage(
          familyPayload,
        )}</span>`;
      }
      tr.appendChild(cell);
    });

    tbody.appendChild(tr);
  });

  const current = currentView();
  elements.statusLine.textContent = `${current.selected_run_count} runs selected, ${rows.length} row combinations visible, ${visibleMetrics.length} metrics shown.`;
};

const render = () => {
  if (state.sortMetricId && !state.visibleMetricIds.has(state.sortMetricId)) {
    state.sortMetricId = null;
    state.sortDirection = "desc";
  }
  renderTable();
  createSegmentedControl();
};

const applyDefaultState = () => {
  state.model = bundle.default_state.model;
  state.dataset = bundle.default_state.dataset;
  state.aggregationMode = bundle.default_state.aggregation_mode;
  state.visibleMetricIds = new Set(bundle.default_state.visible_metric_ids);
  state.visibleProbeTrainOn = new Set(bundle.default_state.visible_probe_train_on);
  state.visibleEvalOn = new Set(bundle.default_state.visible_eval_on);
  state.sortMetricId = null;
  state.sortFieldId = null;
  state.sortDirection = "desc";
};

const bindCoreEvents = () => {
  elements.modelSelect.addEventListener("change", (event) => {
    state.model = event.target.value;
    render();
  });
  elements.datasetSelect.addEventListener("change", (event) => {
    state.dataset = event.target.value;
    render();
  });
  elements.resetSortButton.addEventListener("click", () => {
    state.sortMetricId = null;
    state.sortFieldId = null;
    state.sortDirection = "desc";
    render();
  });
};

const renderError = () => {
  const template = document.getElementById("error-template");
  document.body.innerHTML = "";
  document.body.appendChild(template.content.cloneNode(true));
};

const initialize = async () => {
  try {
    const response = await fetch(BUNDLE_PATH);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    bundle = await response.json();
  } catch (error) {
    console.error(error);
    renderError();
    return;
  }

  applyDefaultState();
  renderControls();
  bindCoreEvents();
  render();
};

initialize();
