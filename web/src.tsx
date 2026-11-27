import { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Plotly from 'plotly.js-dist-min';
import './style.css';

type Series = (number | null)[];
type Model = 'temporal' | 'single_window';
type Recording = {
  start: number; stop: number; initial: number; sampling_rate_hz: number;
  missing_percent: number | null; has_raw: boolean; temporal_role: string | null;
  temporal_available: boolean; temporal_error?: string | null; recorded: { time: number[]; value: Series };
};
type Signal = {
  time: number[]; raw: Series | null; filtered: Series;
  spectrum: { frequency: number[]; power: Series; entropy: number | null; contiguous_seconds: number;
    bands: { name: string; low: number; high: number; power: number }[] } | null;
};
type Predictions = { time: number[]; recorded: Series; predicted: Series; source: string; hop_sec?: number };
type MetricRow = Record<string, number | null>;
type Detail = Record<string, string | number | null>;
type Study = {
  models: Record<string, MetricRow>; test_cases: number[]; history_seconds?: number;
  n_test_windows?: number; n_windows?: number; ranges?: Detail[]; transitions?: Detail[];
};
type PointEvent = { points: { x: number }[] };
type ZoomEvent = { 'xaxis.range[0]'?: number; 'xaxis.range[1]'?: number; 'xaxis.autorange'?: boolean };
type PlotNode = HTMLDivElement & {
  on?: {
    (event: 'plotly_click', handler: (value: PointEvent) => void): void;
    (event: 'plotly_relayout', handler: (value: ZoomEvent) => void): void;
  };
  removeListener?: PlotNode['on'];
};

const colors = { eeg: '#008574' };
const monitorColors = { raw: '#8199a9', eeg: '#54e3a5', actual: '#6bddfa', predicted: '#ffc879' };
const modelNames: Record<string, string> = {
  temporal: 'Temporal CNN', single_window: 'Single-window CNN', random_forest: 'Random Forest',
  cnn: 'Original 1D CNN', median: 'Median baseline', hist_gradient_boosting: 'Histogram Gradient Boosting',
};
const number = (value: number | null | undefined, digits = 1) => value == null ? '—' : value.toFixed(digits);
const clock = (value: number) => {
  const seconds = Math.max(0, Math.floor(value));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((part) => String(part).padStart(2, '0')).join(':');
};

async function request<T>(route: string, signal?: AbortSignal, method = 'GET'): Promise<T> {
  const response = await fetch('/api' + route, { method, signal });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === 'string' ? error.detail : 'The local data request failed.');
  }
  return response.json() as Promise<T>;
}

function Plot({ data, title, unit, range, cursor, onPoint, onZoom, height = 240, bis = false, monitor = false }: {
  data: object[]; title: string; unit: string; range?: [number, number];
  cursor?: number; onPoint?: (x: number) => void; onZoom?: (range: [number, number] | null) => void; height?: number; bis?: boolean; monitor?: boolean;
}) {
  const ref = useRef<PlotNode>(null);
  const click = useRef(onPoint);
  const latestCursor = useRef(cursor);
  const zoom = useRef(onZoom);
  zoom.current = onZoom;
  click.current = onPoint;
  latestCursor.current = cursor;
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const node = ref.current!;
    const observer = new ResizeObserver(() => {
      if (node.clientWidth > 0) void Plotly.relayout(node, { width: node.clientWidth }).catch(() => {});
    });
    observer.observe(node);
    return () => { observer.disconnect(); Plotly.purge(node); };
  }, []);

  useEffect(() => {
    const node = ref.current!;
    let active = true;
    const handle = (event: PointEvent) => {
      const x = event.points[0]?.x;
      if (typeof x === 'number') click.current?.(x);
    };
    const handleZoom = (event: ZoomEvent) => {
      const left = event['xaxis.range[0]'];
      const right = event['xaxis.range[1]'];
      if (typeof left === 'number' && typeof right === 'number') zoom.current?.([left, right]);
      else if (event['xaxis.autorange']) zoom.current?.(null);
    };
    setReady(false);
    void Plotly.react(node, data, {
      height, width: node.clientWidth, margin: { t: 10, r: 14, b: 33, l: 56 },
      paper_bgcolor: monitor ? '#0a141c' : '#fff', plot_bgcolor: monitor ? '#0a141c' : '#fff', showlegend: false,
      font: { family: 'Inter, system-ui, sans-serif', size: 11, color: monitor ? '#a5bac8' : '#526872' },
      xaxis: { range, gridcolor: monitor ? '#213541' : '#e9eff1', zeroline: false, fixedrange: false },
      yaxis: { title: { text: unit }, gridcolor: monitor ? '#213541' : '#e9eff1', zerolinecolor: monitor ? '#344d5d' : '#cfdee0',
        ...(bis ? { range: [0, 100] } : {}) },
      hovermode: 'closest', dragmode: 'zoom',
      shapes: latestCursor.current == null ? [] :
        [{ type: 'line', x0: latestCursor.current, x1: latestCursor.current,
          y0: 0, y1: 1, yref: 'paper', line: { color: monitor ? '#c6dae6' : '#006553', width: 1, dash: 'dot' } }],
    }, { displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] }).then(() => {
      if (active) {
        node.on?.('plotly_click', handle);
        node.on?.('plotly_relayout', handleZoom);
        setReady(true);
      }
    });
    return () => {
      active = false;
      node.removeListener?.('plotly_click', handle);
      node.removeListener?.('plotly_relayout', handleZoom);
    };
  }, [data, unit, range, height, bis, monitor]);

  useEffect(() => {
    if (ready && cursor != null && ref.current) {
      void Plotly.relayout(ref.current, { 'shapes[0].x0': cursor, 'shapes[0].x1': cursor }).catch(() => {});
    }
  }, [ready, cursor]);
  return <div ref={ref} role="img" aria-label={title} style={{ minHeight: height }} />;
}

function trace(x: number[], y: Series, name: string, color: string, mode = 'lines', xUnit = 's') {
  return { type: 'scattergl', mode, x, y, name, connectgaps: false, line: { color, width: 1.3 },
    marker: { color, size: 4 }, hovertemplate: '%{x:.2f} ' + xUnit + '<br>%{y:.2f}<extra>' + name + '</extra>' };
}

function nearestIndex(time: number[], value: number) {
  if (!time.length) return -1;
  let low = 0, high = time.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (time[mid] < value) low = mid + 1;
    else high = mid;
  }
  return low > 0 && value - time[low - 1] < time[low] - value ? low - 1 : low;
}

function App() {
  const [cases, setCases] = useState<number[]>([]);
  const [caseId, setCase] = useState<number | null>(null);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [signal, setSignal] = useState<Signal | null>(null);
  const [predictions, setPredictions] = useState<Predictions | null>(null);
  const [model, setModel] = useState<Model>('temporal');
  const [current, setCurrent] = useState(0);
  const [span, setSpan] = useState(30);
  const [speed, setSpeed] = useState(60);
  const [playing, setPlaying] = useState(false);
  const [focused, setFocused] = useState(false);
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null);
  const [waveform, setWaveform] = useState<'both' | 'filtered' | 'raw'>('both');
  const [signalView, setSignalView] = useState<'psd' | 'bands'>('psd');
  const [relative, setRelative] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [signalError, setSignalError] = useState('');
  const pending = useRef<AbortController | null>(null);

  useEffect(() => {
    request<number[]>('/cases').then((data) => {
      setCases(data);
      setCase(data[0] ?? null);
      if (!data.length) setError('No local recordings are available. Run VitalDB data preparation first.');
    }).catch((failure: Error) => setError(failure.message));
  }, []);

  useEffect(() => {
    if (caseId == null) return;
    const abort = new AbortController();
    pending.current?.abort();
    setBusy(false); setError(''); setRecording(null); setSignal(null); setPredictions(null); setPlaying(false); setZoomRange(null);
    request<Recording>('/cases/' + caseId, abort.signal).then((data) => {
      setRecording(data);
      setCurrent(data.initial);
      setWaveform(data.has_raw ? 'both' : 'filtered');
      setModel(data.temporal_available && !['Training case', 'Validation case'].includes(data.temporal_role ?? '')
        ? 'temporal' : 'single_window');
    }).catch((failure: Error) => { if (!abort.signal.aborted) setError(failure.message); });
    return () => abort.abort();
  }, [caseId]);

  const start = recording?.start ?? 0;
  const end = recording?.stop ?? 1;
  const activeZoom = zoomRange && current >= zoomRange[0] && current <= zoomRange[1] ? zoomRange : null;
  const clipStart = activeZoom?.[0] ?? Math.max(start, Math.min(end - span, start + Math.floor((current - start) / span) * span));
  const clipEnd = activeZoom?.[1] ?? Math.min(end, clipStart + span);
  const eegRange = useMemo<[number, number]>(() =>
    signal?.time.length ? [signal.time[0], signal.time[signal.time.length - 1]] : [clipStart, clipEnd],
  [signal, clipStart, clipEnd]);
  const bisRange = useMemo<[number, number]>(() => activeZoom ?? (focused ? eegRange : [start, end]), [activeZoom, focused, eegRange, start, end]);

  useEffect(() => {
    if (!recording || caseId == null) return;
    const abort = new AbortController();
    setSignalError('');
    const timer = window.setTimeout(() => {
      request<Signal>('/cases/' + caseId + '/signal?start=' + clipStart + '&end=' + clipEnd, abort.signal)
        .then((payload) => { if (!abort.signal.aborted) setSignal(payload); })
        .catch((failure: Error) => { if (!abort.signal.aborted) { setSignal(null); setSignalError(failure.message); } });
    }, 90);
    return () => { window.clearTimeout(timer); abort.abort(); };
  }, [caseId, recording, clipStart, clipEnd]);

  useEffect(() => {
    if (!playing) return;
    let tick = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsed = (now - tick) / 1000;
      tick = now;
      setCurrent((time) => Math.min(end, time + elapsed * speed));
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, end, speed]);
  useEffect(() => { if (playing && current >= end) setPlaying(false); }, [current, end, playing]);
  useEffect(() => {
    if (zoomRange && (current < zoomRange[0] || current > zoomRange[1])) setZoomRange(null);
  }, [current, zoomRange]);

  function zoomTimeline(range: [number, number] | null) {
    if (!range) {
      setZoomRange(null);
      return;
    }
    const left = Math.max(start, range[0]), right = Math.min(end, range[1]);
    if (right - left < 0.5) return;
    setZoomRange([left, right]);
    setCurrent((time) => Math.min(right, Math.max(left, time)));
    setFocused(true);
  }

  async function infer() {
    pending.current?.abort();
    const abort = new AbortController();
    pending.current = abort;
    setBusy(true); setError('');
    try {
      const result = await request<Predictions>('/cases/' + caseId + '/predict?model=' + model, abort.signal, 'POST');
      if (!abort.signal.aborted) setPredictions(result);
    } catch (failure) {
      if (!abort.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!abort.signal.aborted) setBusy(false);
    }
  }

  const eegData = useMemo(() => {
    if (!signal) return [];
    return [
      ...(signal.raw && waveform !== 'filtered' ? [trace(signal.time, signal.raw, 'Raw EEG', monitorColors.raw)] : []),
      ...(waveform !== 'raw' ? [trace(signal.time, signal.filtered, 'Filtered EEG', monitorColors.eeg)] : []),
    ];
  }, [signal, waveform]);

  const bisData = useMemo(() => [
    ...(recording ? [trace(recording.recorded.time, recording.recorded.value, 'Recorded BIS', monitorColors.actual, 'markers')] : []),
    ...(predictions ? [trace(predictions.time, predictions.predicted, 'Predicted BIS', monitorColors.predicted, 'markers')] : []),
  ], [recording, predictions]);

  const spectrum = signal?.spectrum;
  const spectrumData = useMemo(() => spectrum ?
    signalView === 'bands' ? [{ type: 'bar', x: spectrum.bands.map((band) => band.name),
      y: spectrum.bands.map((band) => {
        const total = spectrum.bands.reduce((sum, item) => sum + item.power, 0);
        return relative && total > 0 ? 100 * band.power / total : band.power;
      }), marker: { color: colors.eeg },
      customdata: spectrum.bands.map((band) => band.low + '–' + band.high + ' Hz'),
      hovertemplate: '%{x} · %{customdata}<br>%{y:.3g}' + (relative ? '%' : ' µV²') + '<extra></extra>' }] :
      [trace(spectrum.frequency, spectrum.power, 'Welch PSD', colors.eeg, 'lines', 'Hz')] : [],
  [spectrum, signalView, relative]);

  const nearest = predictions ? nearestIndex(predictions.time, current) : -1;
  const index = predictions && nearest >= 0 &&
    Math.abs(predictions.time[nearest] - current) <= (predictions.hop_sec ?? 5) / 2 ? nearest : -1;
  const predicted = predictions && index >= 0 ? predictions.predicted[index] : null;
  const actual = predictions && index >= 0 ? predictions.recorded[index] : null;
  const monitorIndex = recording ? nearestIndex(recording.recorded.time, current) : -1;
  const monitorReading = recording && monitorIndex >= 0 &&
    Math.abs(recording.recorded.time[monitorIndex] - current) <= 2.5 ? recording.recorded.value[monitorIndex] : null;
  const displayedRecorded = index >= 0 ? actual : monitorReading;
  const predictionTime = predictions && index >= 0 ? predictions.time[index] : null;
  const developmentCase = model === 'temporal' && ['Training case', 'Validation case'].includes(recording?.temporal_role ?? '');

  function exportCsv() {
    if (!predictions) return;
    const csv = 'window_end_sec,recorded_bis,predicted_bis\n' + predictions.time.map((time, i) =>
      [time, predictions.recorded[i] ?? '', predictions.predicted[i] ?? ''].join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url; link.download = 'case-' + caseId + '-' + model + '.csv'; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <>
    <header className="topbar"><div className="shell top-content">
      <a className="brand" href="#top"><span className="mark" aria-hidden="true">∿</span> EEG Research Monitor</a>
      <nav aria-label="Dashboard sections"><a href="#recording">Recording</a><a href="#analysis">Signal analysis</a><a href="#performance">Performance</a></nav>
    </div></header>
    <main className="shell" id="top">
      <section className="intro">
        <div><h1>EEG &amp; BIS Monitor</h1>
          <p>VitalDB surgical recordings · Research and education</p></div>
        <span className="mode-label">Offline replay</span>
      </section>

      <section id="recording" aria-label="Recording workspace" className="monitor">
        <div className="recording-bar panel">
          <div className="recording-selector"><label>VitalDB recording<select value={caseId ?? ''} onChange={(event) => {
            pending.current?.abort();
            setCase(Number(event.target.value));
          }}><option disabled value="">Choose a case</option>{cases.map((id) =>
            <option key={id} value={id}>Case {String(id).padStart(5, '0')}</option>)}</select></label>
            {recording && caseId != null && <a className="dataset-download"
              href={'/api/cases/' + caseId + '/dataset'} download>Download case dataset (ZIP)</a>}
          </div>
          <div><strong>{recording ? clock(end - start) : '—'}</strong><span>EEG duration</span></div>
          <div><strong>{number(recording?.sampling_rate_hz, 0)} Hz</strong><span>Sampling rate</span></div>
          <div><strong>{recording?.missing_percent == null ? '—' : number(recording.missing_percent) + '%'}</strong><span>Raw EEG missing</span></div>
        </div>
        {error && <p role="alert" className="notice error">{error}</p>}
        {!recording && caseId != null && !error && <p role="status" className="notice">Loading recording…</p>}
        {recording && <div className="workspace">
          <section className="panel charts" aria-labelledby="timeline-title">
            <div className="panel-heading"><div><h2 id="timeline-title">Waveforms</h2>
              <p>Scrub to inspect. Drag across a chart to zoom.</p></div>
              <button className="quiet" aria-pressed={focused} onClick={() => { setFocused(!focused); setZoomRange(null); }}>{focused ? 'Show full BIS timeline' : 'Link chart ranges'}</button>
            </div>
            <div className="transport">
              <button className="primary" onClick={() => {
                if (current >= end) setCurrent(start);
                setPlaying(!playing);
              }}>{playing ? 'Pause' : 'Play'}</button>
              <label className="seek"><span>{clock(current - start)} <small>/ {clock(end - start)}</small></span>
                <input aria-label="Recording playhead" type="range" min={start} max={end} step="0.1" value={current}
                  onChange={(event) => setCurrent(Number(event.target.value))} /></label>
              <label>Speed<select value={speed} onChange={(event) => setSpeed(+event.target.value)}>
                {[1, 10, 30, 60, 120, 300].map((factor) => <option key={factor} value={factor}>{factor}×</option>)}</select></label>
            </div>
            <div className="chart-head"><div><h3 className="eeg-channel">EEG</h3><p>Scalp voltage · µV</p></div>
              <div className="chart-controls">
                <select aria-label="EEG traces" value={waveform} onChange={(event) => setWaveform(event.target.value as typeof waveform)}>
                  {recording.has_raw && <><option value="both">Raw + filtered</option><option value="raw">Raw only</option></>}
                  <option value="filtered">Filtered only</option></select>
                <select aria-label="EEG interval" value={span} onChange={(event) => { setSpan(+event.target.value); setZoomRange(null); }}>
                  {[10, 30, 60, 120].map((seconds) => <option key={seconds} value={seconds}>{seconds} s</option>)}</select>
              </div>
            </div>
            {signalError ? <p className="notice error">{signalError}</p> :
              <Plot data={eegData} title="EEG waveform" unit="µV" monitor height={204} range={eegRange} cursor={current} onPoint={setCurrent} onZoom={zoomTimeline} />}
            {!recording.has_raw && <p className="footnote">Cached filtered EEG only. The nearest accepted window is shown.</p>}
            <p className="axis-note">Recording time (s)</p>
            <div className="chart-head bis-head"><div><h3 className="bis-channel">BIS trend</h3><p>Monitor index · 0–100</p></div>
              <div className="legend"><span><i style={{ background: monitorColors.actual }} />Recorded</span>
                {predictions && <span><i style={{ background: monitorColors.predicted }} />Predicted</span>}</div></div>
            <Plot data={bisData} title="Recorded and predicted BIS" unit="BIS" bis monitor height={204} range={bisRange} cursor={current} onPoint={setCurrent} onZoom={zoomTimeline} />
            <p className="axis-note">Recording time (s)</p>
          </section>
          <aside className="panel prediction" aria-labelledby="prediction-title">
            <h2 id="prediction-title">BIS readings</h2>
            <div className="reading recorded-reading">
              <span>Recorded BIS</span><strong>{number(displayedRecorded)}</strong>
              <small>{index >= 0 ? 'Monitor median for selected window' : 'Monitor reading near playhead'}</small>
            </div>
            <div className="reading predicted-reading">
              <span>Estimated BIS</span><strong>{number(predicted)}</strong>
              <small>EEG model estimate</small>
            </div>
            <div className="error-reading"><span>Estimate error</span><strong>{predicted == null || actual == null ? '—' :
              (predicted - actual > 0 ? '+' : '') + number(predicted - actual) + ' pts'}</strong></div>
            <label>Prediction model<select value={model} onChange={(event) => {
              pending.current?.abort(); setBusy(false); setError(''); setPredictions(null);
              setModel(event.target.value as Model);
            }}>
              {recording.temporal_available && <option value="temporal">Temporal CNN</option>}
              <option value="single_window">Single-window CNN</option></select></label>
            <p className={'model-role' + (developmentCase ? ' caution' : '')}>{model === 'temporal'
              ? recording.temporal_role : 'Phase 4 model'}</p>
            {recording.temporal_error && <p className="footnote" role="status">{recording.temporal_error}</p>}
            <p className="model-note">{model === 'temporal'
              ? developmentCase ? 'This case contributed to model development. Use its predictions for exploration.'
                : 'Uses recent EEG history and resets across gaps.'
              : 'Estimates BIS independently for each usable EEG window.'}</p>
            <button className="primary wide" onClick={infer} disabled={busy}>{busy ? 'Estimating BIS…' : 'Run inference'}</button>
            <p className="footnote">{predictionTime == null ? predictions
              ? 'No estimate near this playhead position. Click a predicted BIS point to inspect its window.'
              : 'Run inference to compare recorded and predicted BIS near the playhead.'
              : 'Nearest window ends at ' + predictionTime.toFixed(2) + ' s. Error is predicted minus recorded BIS.'}</p>
            {predictions && <><p className="footnote">{predictions.source}</p>
              <button className="quiet wide" onClick={exportCsv}>Download predictions CSV</button></>}
          </aside>
        </div>}
      </section>

      <section className="panel analysis" id="analysis"><div className="panel-heading"><div>
        <h2>Signal analysis</h2><p>Explore the strongest rhythms in the EEG interval shown above.</p></div>
        <div className="switch" aria-label="Spectrum view">
          <button aria-pressed={signalView === 'psd'} onClick={() => setSignalView('psd')}>Welch spectrum</button>
          <button aria-pressed={signalView === 'bands'} onClick={() => setSignalView('bands')}>Band power</button>
        </div></div>
        <div className="analysis-grid"><div>
          {signalView === 'bands' && <label className="band-scale">Band power scale<select value={relative ? 'relative' : 'absolute'}
            onChange={(event) => setRelative(event.target.value === 'relative')}>
            <option value="absolute">Absolute</option><option value="relative">Relative</option>
          </select></label>}
          {spectrum ? <><Plot data={spectrumData} title="EEG power by frequency" unit={signalView === 'psd' ? 'µV²/Hz' : relative ? 'Share (%)' : 'µV²'} />
            {signalView === 'psd' && <p className="axis-note">Frequency (Hz)</p>}</> :
            <p className="notice">Choose an interval with usable EEG to see its frequency content.</p>}
          <p className="footnote">{signalView === 'psd' ? 'Welch’s method averages overlapping spectra to show where power is concentrated.' :
            'Band power sums the spectrum within the delta, theta, alpha, beta, and gamma ranges.'}</p>
        </div><div className="entropy">
          <h3>Spectral entropy</h3><strong>{number(spectrum?.entropy, 3)}</strong>
          <p>Near 0 means power is concentrated in a few frequencies. Near 1 means power is more evenly spread.</p>
          <p>This is one of the engineered features used by Random Forest.</p>
          {spectrum && <small>Computed from {number(spectrum.contiguous_seconds)} s of contiguous EEG.</small>}
        </div></div>
      </section>

      <Performance />
      <footer>Research and education only · Recorded surgical data · <a href="https://physionet.org/content/vitaldb/" target="_blank" rel="noreferrer">VitalDB dataset</a></footer>
    </main>
  </>;
}

const metricText: Record<string, [string, string]> = {
  case_averaged_mae: ['Case-averaged MAE', 'Compute each case’s mean absolute error, then weight all cases equally. Lower is better.'],
  mae: ['Pooled MAE', 'Average absolute error over all test windows, in BIS points. Lower is better.'],
  rmse: ['RMSE', 'Measures error in BIS points and gives more weight to large misses. Lower is better.'],
  r2: ['R²', 'Compares squared error with a constant predictor equal to the test mean. Higher is better.'],
  pearson_r: ['Pearson r', 'Measures how closely predictions and readings move together linearly.'],
};

function Performance() {
  const [studies, setStudies] = useState<Record<string, Study>>({});
  const [selected, setSelected] = useState('expanded');
  const [metric, setMetric] = useState('case_averaged_mae');
  const [error, setError] = useState('');
  useEffect(() => {
    request<Record<string, Study>>('/studies').then((data) => {
      setStudies(data);
      if (!data.expanded) { setSelected('pilot'); setMetric('mae'); }
    }).catch((failure: Error) => setError(failure.message));
  }, []);
  const study = studies[selected];
  const models = Object.entries(study?.models ?? {});
  const fields = Object.keys(metricText).filter((key) => models.some(([, row]) => key in row));
  const metricData = useMemo(() => [{ type: 'bar', x: models.map(([name]) => modelNames[name] ?? name),
    y: models.map(([, row]) => row[metric]), marker: { color: models.map(([name]) => name === 'temporal' ? colors.eeg : '#77949d') } }],
  [study, metric]);
  return <section className="panel performance" id="performance">
    <div className="panel-heading"><div><h2>Model performance</h2><p>Compare models evaluated on the same held-out cases within each study.</p></div>
      {study && <label>Evaluation study<select value={selected} onChange={(event) => {
        setSelected(event.target.value); setMetric(event.target.value === 'expanded' ? 'case_averaged_mae' : 'mae');
      }}>{Object.keys(studies).map((key) => <option key={key} value={key}>{key === 'expanded' ? 'Expanded temporal study' : 'Original Phase 4 pilot'}</option>)}</select></label>}</div>
    {error && <p role="alert" className="notice error">{error}</p>}
    {study ? <>
      <p className="study-meta">Test cases {study.test_cases.join(', ')} · {(study.n_test_windows ?? study.n_windows)?.toLocaleString()} test windows
        {study.history_seconds && ' · ' + study.history_seconds + ' s temporal context'}</p>
      <div className="metrics"><label>Metric<select value={metric} onChange={(event) => setMetric(event.target.value)}>
        {fields.map((key) => <option key={key} value={key}>{metricText[key][0]}</option>)}</select></label>
        <p>{metricText[metric][1]}</p></div>
      <Plot data={metricData} title="Held-out model comparison" unit={metricText[metric][0]} />
      <div className="table-wrap"><table><thead><tr><th scope="col">Model</th>{fields.map((key) => <th scope="col" key={key}>{metricText[key][0]}</th>)}</tr></thead>
        <tbody>{models.map(([name, row]) => <tr key={name}><th scope="row">{modelNames[name] ?? name}</th>
          {fields.map((key) => <td key={key}>{number(row[key], 2)}</td>)}</tr>)}</tbody></table></div>
      <details><summary>Error by recorded BIS range</summary><DetailTable rows={study.ranges}
        fields={{ model: 'Model', bis_range: 'BIS range', n_windows: 'Windows', mae: 'MAE', bias: 'Bias' }} /></details>
      {selected === 'expanded' && <details><summary>Transition tracking</summary>
        <p>Lag is measured only for matched same-direction crossings of BIS 60 within 30 seconds. Check the matched counts alongside lag.</p>
        <DetailTable rows={study.transitions} fields={{ model: 'Model', case_id: 'Case', reference_crossings: 'Recorded crossings',
          matched_crossings: 'Matched', median_signed_lag_sec: 'Median signed lag (s)' }} /></details>}
    </> : <p className="notice">Evaluation results appear here after model training writes the reports.</p>}
  </section>;
}

function DetailTable({ rows, fields }: { rows?: Detail[]; fields: Record<string, string> }) {
  if (!rows?.length) return <p>This breakdown is unavailable.</p>;
  return <div className="table-wrap"><table><thead><tr>{Object.entries(fields).map(([key, label]) =>
    <th scope="col" key={key}>{label}</th>)}</tr></thead><tbody>
    {rows.map((row, i) => <tr key={i}>{Object.keys(fields).map((key) => {
      const value = row[key];
      return <td key={key}>{typeof value === 'number' ? number(value, Number.isInteger(value) ? 0 : 2) :
        key === 'model' && value ? modelNames[value] ?? value : value ?? '—'}</td>;
    })}</tr>)}
  </tbody></table></div>;
}

createRoot(document.getElementById('root')!).render(<App />);
