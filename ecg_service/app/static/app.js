const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file');
const headerFileInput = document.querySelector('#header-file');
const customRate = document.querySelector('#custom-rate');
const removeHeaderButton = document.querySelector('#remove-header');
const displayModeInputs = document.querySelectorAll('input[name="display-mode"]');
const displayModeFieldset = document.querySelector('.plot-display-mode');
const statusEl = document.querySelector('#status');
const submit = document.querySelector('#submit');
const summary = document.querySelector('#summary');
const durationEl = document.querySelector('#duration');
const samplesEl = document.querySelector('#samples');
const detectedCountEl = document.querySelector('#detected-count');
const diagnosesEl = document.querySelector('#diagnoses');
const plotEl = document.querySelector('#plot');
const DEFAULT_ADC_GAIN_PER_MV = 200;
let latestResult = null;
let leadLayoutMeta = [];
let leadSegmentRanges = new Map();

for (const input of displayModeInputs) {
  input.addEventListener('change', () => {
    if (latestResult) {
      renderPlot(
        latestResult.signal.time,
        latestResult.signal.values,
        latestResult.leads,
        latestResult.signal.calibration,
      );
    }
  });
}
headerFileInput.addEventListener('change', updateHeaderMode);
removeHeaderButton.addEventListener('click', () => {
  headerFileInput.value = '';
  updateHeaderMode();
  if (latestResult) {
    renderPlot(
      latestResult.signal.time,
      latestResult.signal.values,
      latestResult.leads,
      latestResult.signal.calibration,
    );
  }
});
updateHeaderMode();

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus('Choose a signal file first.', 'error');
    return;
  }

  const hasHeader = hasHeaderFile();

  const rate = hasHeader ? 500 : Number(customRate.value);
  if (!Number.isFinite(rate) || rate <= 0) {
    setStatus('Enter a positive custom sampling rate.', 'error');
    return;
  }

  const body = new FormData();
  body.append('file', file);
  if (headerFileInput.files[0]) {
    body.append('header_file', headerFileInput.files[0]);
  }
  if (!hasHeader) {
    body.append('sampling_rate', String(Math.round(rate)));
  }

  submit.disabled = true;
  setStatus('Uploading and running prediction...', '');

  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Prediction failed.');
    }
    renderResult(payload);
    displayModeFieldset.classList.toggle('hidden', !hasHeader);
    setStatus('Prediction completed.', 'ok');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    submit.disabled = false;
  }
});

function getDisplayMode() {
  return document.querySelector('input[name="display-mode"]:checked')?.value || 'mv';
}

function hasHeaderFile() {
  return Boolean(headerFileInput.files?.[0]);
}

function updateHeaderMode() {
  const hasHeader = hasHeaderFile();
  customRate.disabled = hasHeader;
  customRate.required = !hasHeader;
  removeHeaderButton.classList.toggle('hidden', !hasHeader);
  
  if (hasHeader) {
    readHeaderSamplingRate(headerFileInput.files[0])
      .then((rate) => {
        if (hasHeaderFile()) {
          customRate.value = rate;
        }
      })
      .catch(() => {
        if (hasHeaderFile()) {
          customRate.value = '';
        }
      });
  } else if (!customRate.value) {
    customRate.value = '500';
  }
}

async function readHeaderSamplingRate(file) {
  const firstLine = (await file.text()).split(/\r?\n/)[0] || '';
  const rate = Number(firstLine.trim().split(/\s+/)[2]);
  if (!Number.isFinite(rate) || rate <= 0) {
    throw new Error('Invalid header sampling rate.');
  }
  return String(Math.round(rate));
}

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = `status ${kind || ''}`;
}

function renderResult(result) {
  latestResult = result;
  summary.classList.remove('hidden');
  durationEl.textContent = `${result.duration_seconds.toFixed(2)} s`;
  samplesEl.textContent = result.samples.toLocaleString();
  detectedCountEl.textContent = result.detected_diagnoses.length.toString();
  renderDiagnoses(result.diagnoses);
  renderPlot(result.signal.time, result.signal.values, result.leads, result.signal.calibration);
}

function renderDiagnoses(diagnoses) {
  diagnosesEl.classList.remove('empty');
  diagnosesEl.innerHTML = '';

  for (const diagnosis of diagnoses) {
    const probability = Math.max(0, Math.min(1, diagnosis.probability));
    const item = document.createElement('div');
    item.className = `diagnosis ${diagnosis.detected ? 'detected' : ''}`;
    const title = diagnosis.name;
    item.innerHTML = `
      <div class="diagnosis-top">
        <span>${escapeHtml(title)}</span>
        <span>${(probability * 100).toFixed(1)}%</span>
      </div>
      <div class="diagnosis-name">${escapeHtml(diagnosis.code)}</div>
      <div class="bar" aria-hidden="true"><span style="width: ${probability * 100}%"></span></div>
    `;
    diagnosesEl.appendChild(item);
  }
}

function renderPlot(time, values, leads, calibration = []) {
  plotEl.classList.remove('empty');
  updateSegmentLengthLabel(null);
  leadLayoutMeta = [];
  leadSegmentRanges = new Map();
  const hasHeaderCalibration = calibration.some(item => item?.source === 'hea');
  const displayMode = hasHeaderCalibration ? getDisplayMode() : 'raw';
  const displayInfo = getDisplayInfo(displayMode);
  const rowGap = 0.012;
  const rowHeight = (1 - rowGap * (leads.length - 1)) / leads.length;
  const layout = {
    margin: { l: 86, r: 20, t: 10, b: 45 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.72)',
    showlegend: false,
    shapes: [],
  };

  const traces = leads.map((lead, index) => {
    const axisNumber = index + 1;
    const axisSuffix = axisNumber === 1 ? '' : axisNumber;
    const domainEnd = 1 - index * (rowHeight + rowGap);
    const domainStart = domainEnd - rowHeight;
    const leadValues = values.map(row => row[index]);
    const leadCalibration = calibration[index] || getDefaultCalibration();
    const tickConfig = getYAxisTickConfig(leadValues, displayMode, leadCalibration);
    leadLayoutMeta.push({ lead, axisSuffix, domainStart, domainEnd });

    if (index > 0) {
      layout.shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'paper',
        x0: 0,
        x1: 1,
        y0: domainEnd + rowGap / 2,
        y1: domainEnd + rowGap / 2,
        line: { color: '#cbd5e1', width: 1 },
      });
    }

    layout[`xaxis${axisSuffix}`] = {
      domain: [0, 1],
      anchor: `y${axisSuffix}`,
      gridcolor: '#e6edf4',
      showticklabels: index === leads.length - 1,
      title: index === leads.length - 1 ? 'Time, s' : '',
      zeroline: false,
    };
    layout[`yaxis${axisSuffix}`] = {
      domain: [domainStart, domainEnd],
      anchor: `x${axisSuffix}`,
      title: { text: displayInfo.unit ? `${lead}, ${displayInfo.unit}` : lead, standoff: 8 },
      gridcolor: '#eef3f8',
      ...tickConfig,
      zeroline: false,
    };

    return {
      x: time,
      y: leadValues,
      customdata: leadValues.map(value => [
        convertYDisplayValue(value, displayMode, leadCalibration),
        formatCalibrationLabel(leadCalibration),
      ]),
      xaxis: `x${axisSuffix}`,
      yaxis: `y${axisSuffix}`,
      type: 'scattergl',
      mode: 'lines',
      name: lead,
      line: { width: 1.2 },
      hovertemplate: displayInfo.unit
        ? `${lead}<br>%{x:.3f}s<br>%{customdata[0]:.4f} ${displayInfo.unit}<br>%{customdata[1]}<extra></extra>`
        : `${lead}<br>%{x:.3f}s<br>%{y:.4f}<extra></extra>`,
    };
  });

  Plotly.newPlot(plotEl, traces, layout, { responsive: true, displaylogo: false }).then(() => {
    attachSegmentLengthLabel();
  });
}

function attachSegmentLengthLabel() {
  if (plotEl.__segmentLengthHandler) {
    plotEl.removeListener('plotly_relayout', plotEl.__segmentLengthHandler);
  }
  if (plotEl.__segmentSelectedHandler) {
    plotEl.removeListener('plotly_selected', plotEl.__segmentSelectedHandler);
  }
  if (plotEl.__segmentDoubleClickHandler) {
    plotEl.removeListener('plotly_doubleclick', plotEl.__segmentDoubleClickHandler);
  }

  plotEl.__segmentLengthHandler = (eventData) => {
    const keys = Object.keys(eventData || {});
    const xRangeChanged = keys.some(key => key.startsWith('xaxis') && (key.includes('.range') || key.includes('.autorange')));
    if (!xRangeChanged) {
      return;
    }

    if (keys.some(key => key.startsWith('xaxis') && key.endsWith('.autorange') && eventData[key])) {
      updateLeadSegmentLabel(getAxisNameFromRelayout(eventData), null);
      return;
    }

    updateLeadSegmentLabel(getAxisNameFromRelayout(eventData), getRelayoutXRange(eventData) || getCurrentXRange(eventData));
  };

  plotEl.__segmentSelectedHandler = (eventData) => {
    const selectedRanges = getSelectedXRangesByLead(eventData);
    if (!selectedRanges.size) {
      const axisName = getAxisNameFromSelection(eventData);
      const range = getSelectionBoxXRange(eventData);
      if (axisName && range) {
        updateLeadSegmentLabel(axisName, range);
      }
      return;
    }

    for (const [leadIndex, range] of selectedRanges) {
      updateLeadSegmentLabelByIndex(leadIndex, range);
    }
  };

  plotEl.__segmentDoubleClickHandler = () => {
    updateSegmentLengthLabel(null);
  };

  plotEl.on('plotly_relayout', plotEl.__segmentLengthHandler);
  plotEl.on('plotly_selected', plotEl.__segmentSelectedHandler);
  plotEl.on('plotly_doubleclick', plotEl.__segmentDoubleClickHandler);
}

function getRelayoutXRange(eventData) {
  const directRangeKey = Object.keys(eventData || {}).find(key => /^xaxis\d*\.range$/.test(key));
  if (directRangeKey && Array.isArray(eventData[directRangeKey])) {
    return normalizeXRange(eventData[directRangeKey][0], eventData[directRangeKey][1]);
  }

  const firstRangeKey = Object.keys(eventData || {}).find(key => /^xaxis\d*\.range\[0\]$/.test(key));
  if (!firstRangeKey) {
    return null;
  }

  const axisName = firstRangeKey.replace('.range[0]', '');
  return normalizeXRange(eventData[`${axisName}.range[0]`], eventData[`${axisName}.range[1]`]);
}

function getAxisNameFromRelayout(eventData) {
  const rangeKey = Object.keys(eventData || {}).find(key => /^xaxis\d*\./.test(key));
  return rangeKey ? rangeKey.split('.')[0] : 'xaxis';
}

function getCurrentXRange(eventData = {}) {
  const rangeKey = Object.keys(eventData).find(key => /^xaxis\d*\./.test(key));
  const axisName = rangeKey ? rangeKey.split('.')[0] : 'xaxis';
  const xAxis = plotEl.layout?.[axisName];
  const range = xAxis?.range;
  if (!Array.isArray(range) || range.length < 2) {
    return null;
  }

  return normalizeXRange(range[0], range[1]);
}

function normalizeXRange(rawStart, rawEnd) {
  const start = Number(rawStart);
  const end = Number(rawEnd);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start === end) {
    return null;
  }

  return [Math.min(start, end), Math.max(start, end)];
}

function getSelectedXRangesByLead(eventData) {
  const valuesByLead = new Map();

  for (const point of eventData?.points || []) {
    const leadIndex = Number(point.curveNumber);
    const x = Number(point.x);
    if (!Number.isInteger(leadIndex) || !Number.isFinite(x)) {
      continue;
    }

    if (!valuesByLead.has(leadIndex)) {
      valuesByLead.set(leadIndex, []);
    }
    valuesByLead.get(leadIndex).push(x);
  }

  const rangesByLead = new Map();
  for (const [leadIndex, xValues] of valuesByLead) {
    rangesByLead.set(leadIndex, normalizeXRange(Math.min(...xValues), Math.max(...xValues)));
  }

  return rangesByLead;
}

function getAxisNameFromSelection(eventData) {
  const firstPoint = eventData?.points?.[0];
  if (firstPoint && Number.isInteger(Number(firstPoint.curveNumber))) {
    const meta = leadLayoutMeta[Number(firstPoint.curveNumber)];
    return meta ? getLayoutXAxisName(meta.axisSuffix) : null;
  }

  return null;
}

function getSelectionBoxXRange(eventData) {
  if (eventData?.range?.x) {
    return normalizeXRange(eventData.range.x[0], eventData.range.x[1]);
  }
  return null;
}

function updateSegmentLengthLabel(range) {
  if (!range) {
    leadSegmentRanges.clear();
  } else {
    for (let index = 0; index < leadLayoutMeta.length; index += 1) {
      leadSegmentRanges.set(index, range);
    }
  }
  updatePerLeadSegmentAnnotations();
}

function updateLeadSegmentLabel(axisName, range) {
  const leadIndex = leadLayoutMeta.findIndex(meta => getLayoutXAxisName(meta.axisSuffix) === axisName);
  updateLeadSegmentLabelByIndex(leadIndex, range);
}

function getLayoutXAxisName(axisSuffix) {
  return `xaxis${axisSuffix}`;
}

function getAnnotationXAxisRef(axisSuffix) {
  return `x${axisSuffix}`;
}

function updateLeadSegmentLabelByIndex(leadIndex, range) {
  if (leadIndex < 0 || leadIndex >= leadLayoutMeta.length) {
    return;
  }

  if (range) {
    leadSegmentRanges.set(leadIndex, range);
  } else {
    leadSegmentRanges.delete(leadIndex);
  }

  updatePerLeadSegmentAnnotations();
}

function updatePerLeadSegmentAnnotations() {
  if (!plotEl.layout) {
    return;
  }

  const annotations = (plotEl.layout.annotations || [])
    .filter(annotation => !annotation.name?.startsWith('segment-length-'));

  for (const [leadIndex, range] of leadSegmentRanges) {
    const meta = leadLayoutMeta[leadIndex];
    if (!meta || !range) {
      continue;
    }

    const segmentLength = formatSegmentLength(Math.abs(range[1] - range[0]));
    const x = (range[0] + range[1]) / 2;
    annotations.push({
      name: `segment-length-${meta.lead}`,
      xref: getAnnotationXAxisRef(meta.axisSuffix),
      yref: 'paper',
      x,
      y: meta.domainEnd - 0.012,
      xanchor: 'center',
      yanchor: 'top',
      text: `${meta.lead}: ${segmentLength}`,
      showarrow: false,
      font: { size: 11, color: '#0b5953' },
      bgcolor: 'rgba(240, 253, 250, 0.96)',
      bordercolor: '#14b8a6',
      borderwidth: 1,
      borderpad: 4,
    });
  }

  requestAnimationFrame(() => {
    Plotly.relayout(plotEl, { annotations });
  });
}

function formatSegmentLength(seconds) {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)} ms`;
  }
  if (seconds < 10) {
    return `${seconds.toFixed(3)} s`;
  }
  return `${seconds.toFixed(2)} s`;
}

function getDisplayInfo(displayMode) {
  if (displayMode === 'raw') {
    return { unit: '' };
  }
  if (displayMode === 'mm-10') {
    return { unit: 'mm' };
  }
  return { unit: 'mV' };
}

function convertYDisplayValue(value, displayMode, calibration = getDefaultCalibration()) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  if (displayMode === 'raw') {
    return Number(value.toFixed(5));
  }

  const gain = Number(calibration.gain) || DEFAULT_ADC_GAIN_PER_MV;
  const baseline = Number(calibration.baseline) || 0;
  const millivolts = (value - baseline) / gain;
  if (displayMode === 'mv') {
    return Number(millivolts.toFixed(5));
  }

  const displayGainMmPerMv = 10;
  return Number((millivolts * displayGainMmPerMv).toFixed(5));
}

function getDefaultCalibration() {
  return { gain: DEFAULT_ADC_GAIN_PER_MV, baseline: 0, unit: 'mV', source: 'default' };
}

function formatCalibrationLabel(calibration = getDefaultCalibration()) {
  const gain = Number(calibration.gain) || DEFAULT_ADC_GAIN_PER_MV;
  const baseline = Number(calibration.baseline) || 0;
  const source = calibration.source === 'hea' ? '.hea' : 'default';
  return `${source}: gain ${gain}/mV, baseline ${baseline}`;
}

function getYAxisTickConfig(values, displayMode, calibration) {
  const finiteValues = values.filter(Number.isFinite);
  if (!finiteValues.length) {
    return { showticklabels: false, ticks: '' };
  }

  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);
  const range = makeYRange(min, max);
  const tickvals = [range[0], range[1]];
  return {
    range,
    tickmode: 'array',
    tickvals,
    ticktext: tickvals.map(value => formatDisplayTick(value, displayMode, calibration)),
  };
}

function makeYRange(min, max) {
  if (min === max) {
    const padding = Math.max(Math.abs(min) * 0.05, 1);
    return [min - padding, max + padding];
  }

  const padding = (max - min) * 0.05;
  return [
    Number((min - padding).toPrecision(6)),
    Number((max + padding).toPrecision(6)),
  ];
}

function formatDisplayTick(value, displayMode, calibration) {
  const displayValue = convertYDisplayValue(value, displayMode, calibration);
  if (Math.abs(displayValue) >= 100) {
    return displayValue.toFixed(0);
  }
  if (Math.abs(displayValue) >= 10) {
    return displayValue.toFixed(1);
  }
  return displayValue.toFixed(3);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
