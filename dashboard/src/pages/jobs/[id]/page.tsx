
import { useState, useRef, useMemo, useEffect } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft, Play, Square, RotateCcw, Terminal as TerminalIcon,
  Clock, Cpu, Zap, AlertTriangle, CheckCircle, XCircle,
  ChevronRight, BarChart3, Activity, GitBranch, ExternalLink, Loader2,
  Monitor,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line,
} from 'recharts';
import { api, apiClient } from '@/lib/api';

/* ──────── status helpers ──────── */
function StatusIcon({ status }: { status: string }) {
  const s = status.toUpperCase();
  if (s === 'SUCCEEDED') return <CheckCircle style={{ width: 16, height: 16, color: 'var(--success)' }} />;
  if (s === 'RUNNING') return <Play style={{ width: 16, height: 16, color: 'var(--warning)' }} />;
  if (s === 'FAILED' || s === 'ABORTED') return <XCircle style={{ width: 16, height: 16, color: 'var(--error)' }} />;
  return <Clock style={{ width: 16, height: 16, color: 'var(--text-dimmed)' }} />;
}

/* ──────── parse log lines ──────── */
interface LogLine {
  ts: string;
  rank: number;
  level: string;
  msg: string;
}

function parseLogText(raw: string): { logs: LogLine[]; nodeNames: string[] } {
  const lines = raw.split('\n');
  const logs: LogLine[] = [];
  const nodeNames: string[] = [];
  let currentNode = '';

  for (const line of lines) {
    if (line.startsWith('=== Node:')) {
      currentNode = line.replace('=== Node:', '').replace('===', '').trim();
      if (currentNode && !nodeNames.includes(currentNode)) nodeNames.push(currentNode);
      continue;
    }
    if (!line.trim() || line.startsWith('Task:') || line.startsWith('Status:')) continue;

    // Try to parse structured log: timestamp [rank] level: message
    const match = line.match(/^(\d{2}:\d{2}:\d{2})\s+\[?R?(\d+)\]?\s+(\w+)\s+(.+)$/);
    if (match) {
      logs.push({ ts: match[1], rank: parseInt(match[2]), level: match[3], msg: match[4] });
    } else {
      // Raw log line
      logs.push({ ts: '', rank: 0, level: 'INFO', msg: line });
    }
  }

  return { logs, nodeNames };
}

/* ──────── parse training metrics from logs ──────── */
function extractMetrics(logs: LogLine[]) {
  const lossData: { step: number; loss: number; lr: number }[] = [];
  for (const log of logs) {
    // Match: "Step N, Loss: X.XXXX, LR: X.Xe-XX" or similar
    const match = log.msg.match(/Step\s+(\d+).*?Loss:\s+([\d.]+).*?LR:\s+([\d.eE+-]+)/i);
    if (match) {
      lossData.push({
        step: parseInt(match[1]),
        loss: parseFloat(match[2]),
        lr: parseFloat(match[3]),
      });
    }
  }
  return lossData;
}

/* ──────── Pipeline DAG component ──────── */
interface DAGNode { id: string; name: string; status: string; duration: string; }

function PipelineDAG({ nodes, selected, onSelect }: { nodes: DAGNode[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '12px 0' }}>
      {nodes.map((node, i) => (
        <div key={node.id}>
          {i > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', paddingLeft: 26, height: 16 }}>
              <div style={{ width: 2, height: 16, background: node.status === 'pending' ? 'rgba(255,255,255,0.08)' : 'var(--accent-primary)' }} />
            </div>
          )}
          <div
            onClick={() => onSelect(node.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
              cursor: 'pointer', borderRadius: 'var(--radius-sm)',
              background: selected === node.id ? 'rgba(124,58,237,0.12)' : 'transparent',
              border: selected === node.id ? '1px solid rgba(124,58,237,0.3)' : '1px solid transparent',
              transition: 'all 0.15s ease',
            }}
            onMouseOver={e => { if (selected !== node.id) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
            onMouseOut={e => { if (selected !== node.id) e.currentTarget.style.background = selected === node.id ? 'rgba(124,58,237,0.12)' : 'transparent'; }}
          >
            <StatusIcon status={node.status} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{node.name}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dimmed)', fontFamily: 'var(--font-mono)' }}>{node.duration}</div>
            </div>
            {node.status.toUpperCase() === 'RUNNING' && (
              <div style={{
                width: 6, height: 6, borderRadius: '50%', background: 'var(--warning)',
                animation: 'pulse 2s infinite',
              }} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ──────── GPU Gauge component ──────── */
function GpuGauge({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <span style={{ color: 'var(--text-dimmed)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{value.toFixed(0)}%</span>
      </div>
      <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${value}%`, borderRadius: 3,
          background: value > 90 ? 'var(--error)' : value > 70 ? color : 'var(--success)',
          transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  );
}

/* ──────── Main Page ──────── */
export default function JobDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  const jobId = params.id as string;
  const [selectedNode, setSelectedNode] = useState('');
  const [rankFilter, setRankFilter] = useState<number | 'all'>('all');
  const [logSearch, setLogSearch] = useState('');
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Fetch real job data
  const { data: jobData, isLoading: jobLoading } = useQuery({
    queryKey: ['job-detail', jobId],
    queryFn: () => api.jobs.get(jobId),
    retry: 1,
    staleTime: 30_000,
  });

  // Fetch real logs
  const { data: rawLogs } = useQuery({
    queryKey: ['job-logs', jobId],
    queryFn: () => api.jobs.getLogs(jobId),
    retry: 1,
    staleTime: 60_000,
  });

  const job = jobData || {
    job_id: jobId,
    workflow: jobId,
    status: 'UNKNOWN',
    started_at: null,
    gpu_type: null, gpu_count: null,
    domain: null,
  };

  // Parse logs and extract metrics
  const { logs: parsedLogs, nodeNames } = useMemo(() => {
    if (!rawLogs || typeof rawLogs !== 'string') return { logs: [], nodeNames: [] };
    return parseLogText(rawLogs);
  }, [rawLogs]);

  const trainingMetrics = useMemo(() => extractMetrics(parsedLogs), [parsedLogs]);

  // Build DAG nodes from Flyte node names
  const dagNodes: DAGNode[] = useMemo(() => {
    if (nodeNames.length > 0) {
      return nodeNames.map(name => ({
        id: name,
        name: name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        status: job.status || 'UNKNOWN',
        duration: '—',
      }));
    }
    // Fallback: single node from workflow name
    return [{
      id: 'main',
      name: job.workflow || jobId,
      status: job.status || 'UNKNOWN',
      duration: job.duration || '—',
    }];
  }, [nodeNames, job]);

  // Auto-select first node
  useEffect(() => {
    if (!selectedNode && dagNodes.length > 0 && dagNodes[0]) {
      setSelectedNode(dagNodes[0].id);
    }
  }, [dagNodes, selectedNode]);

  const elapsed = job.started_at
    ? (() => {
        const ms = Date.now() - new Date(job.started_at).getTime();
        const mins = Math.floor(ms / 60000);
        const hrs = Math.floor(mins / 60);
        return hrs > 0 ? `${hrs}h ${mins % 60}m` : `${mins}m`;
      })()
    : (job.duration || '—');

  const filteredLogs = parsedLogs.filter(l =>
    (rankFilter === 'all' || l.rank === rankFilter) &&
    (!logSearch || l.msg.toLowerCase().includes(logSearch.toLowerCase()))
  );

  const statusColorMap: Record<string, string> = {
    RUNNING: '#f59e0b', SUCCEEDED: '#10b981', FAILED: '#ef4444',
    ABORTED: '#ef4444', QUEUED: '#64748b',
  };
  const statusColor = statusColorMap[job.status?.toUpperCase?.()] || '#64748b';

  if (jobLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#0a0e1a' }}>
        <Loader2 style={{ width: 24, height: 24, animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />
        <span style={{ marginLeft: 12, color: 'var(--text-muted)' }}>Loading job {jobId}...</span>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0a0e1a' }}>
      {/* Top Header Bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px', height: 52, flexShrink: 0,
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(255,255,255,0.02)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button
            onClick={() => navigate('/jobs')}
            style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none', border: 'none', color: 'var(--text-dimmed)', cursor: 'pointer', fontSize: 13 }}
          >
            <ArrowLeft style={{ width: 14, height: 14 }} /> Jobs
          </button>
          <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.1)' }} />

          {/* Status pulse */}
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: statusColor,
            boxShadow: job.status === 'RUNNING' ? `0 0 8px ${statusColor}` : '',
            animation: job.status === 'RUNNING' ? 'pulse 2s infinite' : '',
          }} />

          <div>
            <div style={{ fontWeight: 600, fontSize: 15 }}>{job.workflow || jobId}</div>
            <div style={{ fontSize: 12, color: 'var(--text-dimmed)', fontFamily: 'var(--font-mono)' }}>{jobId}</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-muted)' }}>
            <Clock style={{ width: 14, height: 14 }} /> {elapsed}
          </div>
          {job.gpu_type && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
              <Cpu style={{ width: 14, height: 14, color: 'var(--accent-primary)' }} />
              <span style={{ fontFamily: 'var(--font-mono)' }}>{job.gpu_type} x{job.gpu_count || 1}</span>
            </div>
          )}
          {job.instance_type && job.instance_type !== 'unknown' && (
            <div style={{ fontSize: 12, color: 'var(--text-dimmed)', fontFamily: 'var(--font-mono)' }}>
              {job.instance_type}
            </div>
          )}
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/experiments')}>
            <BarChart3 style={{ width: 12, height: 12 }} /> Compare Run
          </button>
          <button className="btn btn-ghost btn-sm" style={{ color: 'var(--error)' }}
            title={job.status === 'RUNNING' ? 'Abort this job' : 'Job is not running'}>
            <Square style={{ width: 12, height: 12 }} /> {job.status === 'RUNNING' ? 'Abort' : job.status}
          </button>
        </div>
      </div>

      {/* Error banner */}
      {job.error && (
        <div style={{
          padding: '10px 20px', background: 'rgba(239,68,68,0.08)',
          borderBottom: '1px solid rgba(239,68,68,0.2)', display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 13, color: 'var(--error)',
        }}>
          <AlertTriangle style={{ width: 14, height: 14 }} />
          <span style={{ fontWeight: 600 }}>Error:</span> {job.error}
        </div>
      )}

      {/* Main Content — 3-column layout */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left — Pipeline DAG */}
        <div style={{
          width: 260, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.15)',
        }}>
          <div style={{
            padding: '12px 16px', fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
            letterSpacing: '0.06em', color: 'var(--text-dimmed)',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <GitBranch style={{ width: 14, height: 14 }} /> Pipeline DAG
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px' }}>
            <PipelineDAG nodes={dagNodes} selected={selectedNode} onSelect={setSelectedNode} />
          </div>
        </div>

        {/* Center — Metrics + Logs */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Training Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, borderBottom: '1px solid rgba(255,255,255,0.08)', flexShrink: 0 }}>
            {/* Loss Chart */}
            <div style={{ padding: 16, borderRight: '1px solid rgba(255,255,255,0.08)' }}>
              <div style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-dimmed)', marginBottom: 12 }}>
                Training Loss
              </div>
              <div style={{ height: 160 }}>
                {trainingMetrics.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trainingMetrics}>
                      <defs>
                        <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ background: 'rgba(10,14,26,0.95)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }} />
                      <Area type="monotone" dataKey="loss" stroke="#7c3aed" strokeWidth={2} fill="url(#lossGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-dimmed)', fontSize: 13 }}>
                    {parsedLogs.length > 0 ? 'No training loss metrics in logs' : 'Awaiting log data...'}
                  </div>
                )}
              </div>
            </div>
            {/* Learning Rate Chart */}
            <div style={{ padding: 16 }}>
              <div style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-dimmed)', marginBottom: 12 }}>
                Learning Rate
              </div>
              <div style={{ height: 160 }}>
                {trainingMetrics.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => v.toExponential(0)} />
                      <Tooltip contentStyle={{ background: 'rgba(10,14,26,0.95)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }} />
                      <Line type="monotone" dataKey="lr" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-dimmed)', fontSize: 13 }}>
                    {parsedLogs.length > 0 ? 'No LR schedule in logs' : 'Awaiting log data...'}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Structured Logs */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px',
              borderBottom: '1px solid rgba(255,255,255,0.06)', flexShrink: 0,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-dimmed)' }}>
                Logs
              </div>
              <input
                type="text"
                placeholder="Search logs..."
                value={logSearch}
                onChange={e => setLogSearch(e.target.value)}
                style={{
                  padding: '3px 8px', fontSize: 12, borderRadius: 4, width: 200,
                  background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                  color: 'var(--text-primary)', outline: 'none',
                }}
              />
              <div style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-dimmed)' }}>
                {filteredLogs.length} lines
              </div>
            </div>
            <div style={{
              flex: 1, overflowY: 'auto', padding: '4px 0',
              fontFamily: 'var(--font-mono)', fontSize: 13, lineHeight: 1.65,
              background: '#0d1117',
            }}>
              {filteredLogs.length > 0 ? filteredLogs.map((log, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 8, padding: '1px 16px',
                  background: log.level === 'WARN' ? 'rgba(245,158,11,0.06)' : log.level === 'ERROR' ? 'rgba(239,68,68,0.08)' : 'transparent',
                }}>
                  {log.ts && <span style={{ color: '#64748b', minWidth: 64 }}>{log.ts}</span>}
                  <span style={{
                    color: log.level === 'WARN' ? 'var(--warning)' : log.level === 'ERROR' ? 'var(--error)' : '#c9d1d9',
                  }}>{log.msg}</span>
                </div>
              )) : (
                <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--text-dimmed)', fontSize: 13 }}>
                  {rawLogs === undefined ? 'Loading logs...' : 'No logs available for this execution'}
                </div>
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>

        {/* Right — Job Info */}
        <div style={{
          width: 280, flexShrink: 0, borderLeft: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.1)',
        }}>
          <div style={{
            padding: '12px 16px', fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
            letterSpacing: '0.06em', color: 'var(--text-dimmed)',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <Activity style={{ width: 14, height: 14 }} /> Job Details
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Job Metadata */}
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, color: 'var(--text-muted)' }}>Execution Info</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  ['Workflow', job.workflow],
                  ['Status', job.status],
                  ['Job ID', jobId],
                  ['Duration', job.duration || elapsed],
                  ['Instance', job.instance_type || '—'],
                  ['GPU', job.gpu_type ? `${job.gpu_type} x${job.gpu_count || 1}` : '—'],
                  ['Domain', job.domain || '—'],
                  ['Created', job.created_at ? new Date(job.created_at).toLocaleString() : '—'],
                ].map(([label, value]) => (
                  <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span style={{ color: 'var(--text-dimmed)' }}>{label}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-primary)', textAlign: 'right', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {value || '—'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, color: 'var(--text-muted)' }}>Actions</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <button className="btn btn-ghost btn-sm" style={{ justifyContent: 'flex-start', width: '100%' }}
                  onClick={() => navigate('/experiments')}>
                  <BarChart3 style={{ width: 12, height: 12 }} /> Compare in Experiments
                </button>
                <button className="btn btn-ghost btn-sm" style={{ justifyContent: 'flex-start', width: '100%' }}
                  onClick={() => {
                    const base = import.meta.env.VITE_FLYTE_CONSOLE_URL || '';
                    const consoleUrl = `${base}/console/projects/ml-platform/domains/development/executions/${jobId}`;
                    window.open(consoleUrl, '_blank');
                  }}>
                  <ExternalLink style={{ width: 12, height: 12 }} /> Open in Flyte Console
                </button>
                <button className="btn btn-ghost btn-sm" style={{ justifyContent: 'flex-start', width: '100%' }}
                  onClick={async () => {
                    try {
                      const url = await api.tensorboard.getUrl(jobId);
                      window.open(url, '_blank', 'noopener,noreferrer');
                    } catch {
                      const tbBase = import.meta.env.VITE_TENSORBOARD_URL || 'http://localhost:6006';
                      window.open(`${tbBase}/#scalars&regexInput=${encodeURIComponent(jobId)}`, '_blank', 'noopener,noreferrer');
                    }
                  }}>
                  <Monitor style={{ width: 12, height: 12 }} />
                  {job.status === 'RUNNING' && (
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: 'var(--warning)',
                      animation: 'pulse-dot 2s infinite',
                      display: 'inline-block',
                    }} />
                  )}
                  {job.status === 'RUNNING' ? 'Live TensorBoard' : 'TensorBoard Logs'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
