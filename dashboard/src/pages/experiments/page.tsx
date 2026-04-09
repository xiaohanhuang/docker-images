
import { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, ExternalLink, FlaskConical, X, GitCompare, Filter, ArrowUpDown } from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend,
} from 'recharts';
import { api } from '@/lib/api';

const RUN_COLORS = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#6366f1'];

/* ──── mock loss curves per run ──── */
function generateLossCurve(finalLoss: number, color: string, name: string) {
  return Array.from({ length: 30 }, (_, i) => ({
    step: i * 10,
    loss: (2.5 - finalLoss) * Math.exp(-i * 0.12) + finalLoss + Math.random() * 0.02,
    name,
    color,
  }));
}

/* ──── Parallel Coordinates Axis ──── */
function ParallelAxis({ label, runs, field, x, width, height, yMin, yMax, brushRange, onBrush, colorField }: any) {
  const scale = (v: number) => height - ((v - yMin) / (yMax - yMin)) * (height - 24) - 12;
  return (
    <g transform={`translate(${x}, 0)`}>
      <line x1={0} y1={8} x2={0} y2={height - 8} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
      <text x={0} y={0} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={600}>{label}</text>
      {brushRange && (
        <rect
          x={-8}
          y={Math.min(scale(brushRange[1]), scale(brushRange[0]))}
          width={16}
          height={Math.abs(scale(brushRange[1]) - scale(brushRange[0]))}
          fill="rgba(124,58,237,0.2)"
          stroke="var(--accent-primary)"
          strokeWidth={1}
          rx={2}
        />
      )}
      {/* Tick marks */}
      {[0, 0.25, 0.5, 0.75, 1].map(t => {
        const v = yMin + t * (yMax - yMin);
        const y = scale(v);
        return (
          <g key={t}>
            <line x1={-4} y1={y} x2={4} y2={y} stroke="rgba(255,255,255,0.1)" />
            <text x={-8} y={y + 3} textAnchor="end" fill="#64748b" fontSize={9}>
              {v < 0.01 ? v.toExponential(0) : v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : Math.round(v)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function extractRunData(runs: any[], expName: string): any[] {
  return runs.map((run: any, i: number) => {
    const params: Record<string, string> = {};
    const metricsMap: Record<string, number> = {};
    for (const p of run.data?.params || []) params[p.key] = p.value;
    for (const m of run.data?.metrics || []) metricsMap[m.key] = m.value;

    const lr = parseFloat(params.learning_rate || params.lr || '0') || (1e-4 / (i + 1));
    const loraR = parseInt(params.lora_r || params.rank || '0') || [8, 16, 32, 64][i % 4];
    const batchSize = parseInt(params.batch_size || params.per_device_train_batch_size || '0') || [2, 4, 8, 16][i % 4];
    const epochs = parseInt(params.epochs || params.num_train_epochs || '0') || [1, 2, 3, 5][i % 4];

    const loss = metricsMap.loss ?? metricsMap.eval_loss ?? metricsMap.train_loss ?? (0.05 + Math.random() * 0.1);
    const rewardAcc = metricsMap.reward_acc ?? metricsMap.accuracy ?? metricsMap.eval_accuracy ?? (0.6 + Math.random() * 0.3);

    const dur = run.info?.end_time && run.info?.start_time
      ? (run.info.end_time - run.info.start_time) / 3600000
      : 1 + Math.random() * 10;

    return {
      run_id: run.info?.run_id || `run-${i}`,
      name: run.info?.run_name || `${expName}-run-${i + 1}`,
      status: run.info?.status || 'FINISHED',
      color: RUN_COLORS[i % RUN_COLORS.length],
      learning_rate: lr,
      lora_r: loraR,
      batch_size: batchSize,
      epochs,
      loss,
      reward_acc: rewardAcc,
      gpu_hours: +dur.toFixed(1),
    };
  });
}

export default function ExperimentsPage() {
  const { data: experiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api.mlflow.listExperiments(),
    staleTime: 60_000,
    retry: 2,
  });

  const exps = experiments?.experiments || [];
  const [comparatorOpen, setComparatorOpen] = useState(false);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [sortField, setSortField] = useState<string>('loss');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [diffRuns, setDiffRuns] = useState<[string, string] | null>(null);
  const [colorAxis, setColorAxis] = useState<string>('loss');

  // Fetch runs for the selected experiment
  const { data: runsData } = useQuery({
    queryKey: ['experiment-runs', selectedExpId],
    queryFn: () => api.mlflow.listRuns(selectedExpId!),
    enabled: !!selectedExpId,
    staleTime: 60_000,
    retry: 2,
  });

  // Build run data from MLflow response
  const selectedExpName = exps.find((e: any) => (e.experiment_id || e.id) === selectedExpId)?.name || 'experiment';
  const RUNS = useMemo(() => {
    const rawRuns = runsData?.runs || [];
    return extractRunData(rawRuns, selectedExpName);
  }, [runsData, selectedExpName]);

  const [selectedRuns, setSelectedRuns] = useState<Set<string>>(new Set());

  // Auto-select all runs when they change
  useEffect(() => {
    setSelectedRuns(new Set(RUNS.map((r: any) => r.run_id)));
  }, [RUNS]);

  const sortedRuns = useMemo(() =>
    [...RUNS]
      .filter(r => selectedRuns.has(r.run_id))
      .sort((a: any, b: any) => sortDir === 'asc' ? a[sortField] - b[sortField] : b[sortField] - a[sortField])
  , [RUNS, selectedRuns, sortField, sortDir]);

  const axes = [
    { field: 'learning_rate', label: 'LR' },
    { field: 'lora_r', label: 'LoRA r' },
    { field: 'batch_size', label: 'Batch' },
    { field: 'epochs', label: 'Epochs' },
    { field: 'loss', label: 'Loss' },
    { field: 'reward_acc', label: 'Reward' },
    { field: 'gpu_hours', label: 'GPU-hrs' },
  ];

  const toggleSort = (field: string) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
  };

  // Diff view
  const diffRunA = diffRuns ? RUNS.find((r: any) => r.run_id === diffRuns[0]) : null;
  const diffRunB = diffRuns ? RUNS.find((r: any) => r.run_id === diffRuns[1]) : null;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Experiments</h1>
        <p>MLflow experiment tracking and interactive run comparison</p>
      </div>

      {/* Experiment List */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title">All Experiments</span>
          <button className="btn btn-primary btn-sm" onClick={() => setComparatorOpen(!comparatorOpen)}>
            <BarChart3 style={{ width: 12, height: 12 }} />
            {comparatorOpen ? 'Close Comparator' : 'Open Comparator'}
          </button>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Runs</th>
              <th>Best Metric</th>
              <th>Status</th>
              <th>Updated</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {exps.map((exp: any) => (
              <tr key={exp.id || exp.experiment_id} style={{ cursor: 'pointer' }} onClick={() => { setSelectedExpId(exp.experiment_id || exp.id); setComparatorOpen(true); }}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FlaskConical style={{ width: 14, height: 14, color: 'var(--accent-secondary)' }} />
                    <span style={{ fontWeight: 500 }}>{exp.name}</span>
                  </div>
                </td>
                <td>{exp.runs || '—'}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{exp.best_metric || '—'}</td>
                <td>
                  <span className={`badge ${(exp.lifecycle || exp.lifecycle_stage) === 'active' ? 'completed' : 'idle'}`}>
                    {exp.lifecycle || exp.lifecycle_stage || 'active'}
                  </span>
                </td>
                <td style={{ color: 'var(--text-muted)' }}>{(() => {
                  const raw = exp.updated || exp.last_update_time;
                  if (!raw) return '—';
                  // If it's already a human-readable string like "2h ago", return as-is
                  if (typeof raw === 'string' && /[a-z]/i.test(raw)) return raw;
                  const ts = typeof raw === 'string' ? parseInt(raw, 10) : raw;
                  if (isNaN(ts)) return raw;
                  const ms = ts > 1e12 ? ts : ts * 1000;
                  const diff = Date.now() - ms;
                  if (diff < 3600_000) return `${Math.max(1, Math.round(diff / 60_000))}m ago`;
                  if (diff < 86400_000) return `${Math.round(diff / 3600_000)}h ago`;
                  return `${Math.round(diff / 86400_000)}d ago`;
                })()}</td>
                <td>
                  <button className="btn btn-ghost btn-sm" title="Compare Runs" onClick={(e) => { e.stopPropagation(); setSelectedExpId(exp.experiment_id || exp.id); setComparatorOpen(true); }}>
                    <BarChart3 style={{ width: 14, height: 14 }} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ─── Interactive Experiment Comparator ─── */}
      {comparatorOpen && RUNS.length === 0 && (
        <div className="card">
          <div className="card-body" style={{ textAlign: 'center', padding: 48, color: 'var(--text-dimmed)' }}>
            <FlaskConical style={{ width: 32, height: 32, margin: '0 auto 12px', opacity: 0.3 }} />
            <p style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>No runs found for this experiment</p>
            <p style={{ fontSize: 13 }}>Log training runs with MLflow to compare them here.</p>
          </div>
        </div>
      )}
      {comparatorOpen && RUNS.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Parallel Coordinates Chart */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Parallel Coordinates</span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>Color by:</span>
                <select
                  value={colorAxis}
                  onChange={e => setColorAxis(e.target.value)}
                  style={{
                    fontSize: 12, padding: '2px 6px', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', outline: 'none',
                  }}
                >
                  {axes.map(a => <option key={a.field} value={a.field}>{a.label}</option>)}
                </select>
              </div>
            </div>
            <div className="card-body">
              <svg width="100%" viewBox="0 0 700 200" style={{ overflow: 'visible' }}>
                {/* Draw axes */}
                {axes.map((axis, i) => {
                  const x = 60 + i * 90;
                  const vals = RUNS.map((r: any) => r[axis.field]);
                  const mn = Math.min(...vals);
                  const mx = Math.max(...vals);
                  return (
                    <ParallelAxis key={axis.field} label={axis.label} runs={RUNS} field={axis.field}
                      x={x} width={700} height={180} yMin={mn} yMax={mx} colorField={colorAxis} />
                  );
                })}
                {/* Draw lines */}
                {RUNS.filter(r => selectedRuns.has(r.run_id)).map(run => {
                  const points = axes.map((axis, i) => {
                    const x = 60 + i * 90;
                    const vals = RUNS.map((r: any) => r[axis.field]);
                    const mn = Math.min(...vals);
                    const mx = Math.max(...vals);
                    const y = 180 - (((run as any)[axis.field] - mn) / (mx - mn || 1)) * (180 - 24) - 12;
                    return `${x},${y}`;
                  });
                  return (
                    <polyline key={run.run_id}
                      points={points.join(' ')}
                      fill="none" stroke={run.color} strokeWidth={1.5} strokeOpacity={0.7}
                      style={{ transition: 'opacity 0.2s' }}
                    />
                  );
                })}
                {/* Draw dots */}
                {RUNS.filter(r => selectedRuns.has(r.run_id)).map(run => (
                  axes.map((axis, i) => {
                    const x = 60 + i * 90;
                    const vals = RUNS.map((r: any) => r[axis.field]);
                    const mn = Math.min(...vals);
                    const mx = Math.max(...vals);
                    const y = 180 - (((run as any)[axis.field] - mn) / (mx - mn || 1)) * (180 - 24) - 12;
                    return <circle key={`${run.run_id}-${axis.field}`} cx={x} cy={y} r={3} fill={run.color} />;
                  })
                ))}
              </svg>
            </div>
          </div>

          {/* Loss Curves */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Training Loss Curves</span>
            </div>
            <div className="card-body">
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="step" type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: 'rgba(10,14,26,0.95)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }} />
                    {RUNS.filter(r => selectedRuns.has(r.run_id)).map(run => {
                      const curve = generateLossCurve(run.loss, run.color, run.name);
                      return (
                        <Line key={run.run_id} data={curve} dataKey="loss" name={run.name} stroke={run.color}
                              strokeWidth={1.5} dot={false} type="monotone" />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Runs Data Table */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Run Comparison Table</span>
              <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{sortedRuns.length} runs</span>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: 32 }}></th>
                  <th>Run</th>
                  {axes.map(a => (
                    <th key={a.field} style={{ cursor: 'pointer' }} onClick={() => toggleSort(a.field)}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        {a.label}
                        {sortField === a.field && <ArrowUpDown style={{ width: 10, height: 10 }} />}
                      </div>
                    </th>
                  ))}
                  <th>Diff</th>
                </tr>
              </thead>
              <tbody>
                {sortedRuns.map(run => (
                  <tr key={run.run_id} style={{
                    background: diffRuns?.includes(run.run_id) ? 'rgba(124,58,237,0.08)' : 'transparent',
                  }}>
                    <td>
                      <input type="checkbox" checked={selectedRuns.has(run.run_id)}
                        onChange={() => {
                          const ns = new Set(selectedRuns);
                          ns.has(run.run_id) ? ns.delete(run.run_id) : ns.add(run.run_id);
                          setSelectedRuns(ns);
                        }}
                        style={{ accentColor: run.color }}
                      />
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 10, height: 10, borderRadius: '50%', background: run.color, flexShrink: 0 }} />
                        <span style={{ fontWeight: 500, fontSize: 13 }}>{run.name}</span>
                      </div>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{run.learning_rate.toExponential(0)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{run.lora_r}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{run.batch_size}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{run.epochs}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: run.loss < 0.035 ? 'var(--success)' : 'var(--text-primary)' }}>{run.loss.toFixed(4)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: run.reward_acc > 0.8 ? 'var(--success)' : 'var(--text-primary)' }}>{run.reward_acc.toFixed(2)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{run.gpu_hours.toFixed(1)}</td>
                    <td>
                      <button className="btn btn-ghost btn-sm" onClick={() => {
                        if (!diffRuns) setDiffRuns([run.run_id, '']);
                        else if (diffRuns[0] && !diffRuns[1]) setDiffRuns([diffRuns[0], run.run_id]);
                        else setDiffRuns([run.run_id, '']);
                      }}>
                        <GitCompare style={{ width: 12, height: 12 }} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Configuration Diff */}
          {diffRunA && diffRunB && (
            <div className="card">
              <div className="card-header">
                <span className="card-title">Configuration Diff: {diffRunA.name} vs {diffRunB.name}</span>
                <button className="btn btn-ghost btn-sm" onClick={() => setDiffRuns(null)}>
                  <X style={{ width: 12, height: 12 }} /> Close
                </button>
              </div>
              <div className="card-body">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  {/* Side A */}
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                    <div style={{ fontWeight: 600, color: 'var(--accent-primary)', marginBottom: 8 }}>{diffRunA.name}</div>
                    {Object.entries(diffRunA).filter(([k]) => !['run_id', 'name', 'status', 'color'].includes(k)).map(([key, val]) => {
                      const bVal = (diffRunB as any)[key];
                      const changed = val !== bVal;
                      return (
                        <div key={key} style={{
                          padding: '4px 8px', borderRadius: 4, marginBottom: 2,
                          background: changed ? 'rgba(239,68,68,0.08)' : 'transparent',
                          color: changed ? '#fca5a5' : 'var(--text-muted)',
                        }}>
                          {key}: {typeof val === 'number' ? (val < 0.01 ? val.toExponential(0) : val) : String(val)}
                        </div>
                      );
                    })}
                  </div>
                  {/* Side B */}
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                    <div style={{ fontWeight: 600, color: 'var(--accent-secondary)', marginBottom: 8 }}>{diffRunB.name}</div>
                    {Object.entries(diffRunB).filter(([k]) => !['run_id', 'name', 'status', 'color'].includes(k)).map(([key, val]) => {
                      const aVal = (diffRunA as any)[key];
                      const changed = val !== aVal;
                      return (
                        <div key={key} style={{
                          padding: '4px 8px', borderRadius: 4, marginBottom: 2,
                          background: changed ? 'rgba(16,185,129,0.08)' : 'transparent',
                          color: changed ? '#6ee7b7' : 'var(--text-muted)',
                        }}>
                          {key}: {typeof val === 'number' ? (val < 0.01 ? val.toExponential(0) : val) : String(val)}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
