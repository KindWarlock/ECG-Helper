const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file');
const samplingRate = document.querySelector('#sampling-rate');
const customRate = document.querySelector('#custom-rate');
const rateModeInputs = document.querySelectorAll('input[name="rate-mode"]');
const statusEl = document.querySelector('#status');
const submit = document.querySelector('#submit');
const summary = document.querySelector('#summary');
const durationEl = document.querySelector('#duration');
const samplesEl = document.querySelector('#samples');
const detectedCountEl = document.querySelector('#detected-count');
const diagnosesEl = document.querySelector('#diagnoses');
const plotEl = document.querySelector('#plot');

for (const input of rateModeInputs) {
  input.addEventListener('change', updateRateMode);
}
updateRateMode();

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus('Choose a signal file first.', 'error');
    return;
  }

  const rateMode = getRateMode();
  const rate = Number(rateMode === 'custom' ? customRate.value : samplingRate.value);
  if (!Number.isFinite(rate) || rate <= 0) {
    setStatus(rateMode === 'custom' ? 'Enter a positive custom sampling rate.' : 'Sampling rate must be positive.', 'error');
    return;
  }

  const body = new FormData();
  body.append('file', file);
  body.append('sampling_rate', String(Math.round(rate)));

  submit.disabled = true;
  setStatus('Uploading and running prediction...', '');

  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Prediction failed.');
    }
    renderResult(payload);
    setStatus('Prediction completed.', 'ok');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    submit.disabled = false;
  }
});

function getRateMode() {
  return document.querySelector('input[name="rate-mode"]:checked')?.value || 'predefined';
}

function updateRateMode() {
  const customMode = getRateMode() === 'custom';
  samplingRate.disabled = customMode;
  customRate.disabled = !customMode;
  customRate.required = customMode;
  if (customMode) {
    customRate.focus();
  }
}

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = `status ${kind || ''}`;
}

function renderResult(result) {
  summary.classList.remove('hidden');
  durationEl.textContent = `${result.duration_seconds.toFixed(2)} s`;
  samplesEl.textContent = result.samples.toLocaleString();
  detectedCountEl.textContent = result.detected_diagnoses.length.toString();
  renderDiagnoses(result.diagnoses);
  renderPlot(result.signal.time, result.signal.values, result.leads);
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

function renderPlot(time, values, leads) {
  plotEl.classList.remove('empty');
  const rowGap = 0.012;
  const rowHeight = (1 - rowGap * (leads.length - 1)) / leads.length;
  const layout = {
    margin: { l: 64, r: 20, t: 10, b: 45 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.72)',
    showlegend: false,
  };

  const traces = leads.map((lead, index) => {
    const axisNumber = index + 1;
    const axisSuffix = axisNumber === 1 ? '' : axisNumber;
    const domainEnd = 1 - index * (rowHeight + rowGap);
    const domainStart = domainEnd - rowHeight;

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
      title: { text: lead, standoff: 8 },
      gridcolor: '#eef3f8',
      zeroline: false,
    };

    return {
      x: time,
      y: values.map(row => row[index]),
      xaxis: `x${axisSuffix}`,
      yaxis: `y${axisSuffix}`,
      type: 'scattergl',
      mode: 'lines',
      name: lead,
      line: { width: 1.2 },
      hovertemplate: `${lead}<br>%{x:.3f}s<br>%{y:.4f}<extra></extra>`,
    };
  });

  Plotly.newPlot(plotEl, traces, layout, { responsive: true, displaylogo: false });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
