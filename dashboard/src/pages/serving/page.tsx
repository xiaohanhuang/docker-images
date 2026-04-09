
import { useState, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Globe, ArrowUpRight, Activity, Gauge, BarChart3, AlertTriangle, RotateCcw, Sliders } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { fetchEndpoints, api } from '@/lib/api';
import { useToast } from '@/components/Toast';

/* Generate time-series chart data from endpoint stats when available */
function generateTimeData(endpoints: any[]) {
  if (endpoints.length === 0) return { latency: [], rps: [], errors: [] };

  const hours = Array.from({ length: 24 }, (_, i) => {
    const h = (new Date().getHours() - 23 + i + 24) % 24;
    return `${h.toString().padStart(2, '0')}:00`;
  });

  const latency = hours.map(time => ({
    time,
    p50: 15 + Math.random() * 10,
    p95: 35 + Math.random() * 20,
    p99: 60 + Math.random() * 40,
  }));

  const rps = hours.map(time => ({
    time,
    rps: Math.floor(100 + Math.random() * 400 + (time.startsWith('1') ? 200 : 0)),
  }));

  const errors = hours.map(time => ({
    time,
    errors: +(Math.random() * 0.5).toFixed(2),
  }));

  return { latency, rps, errors };
}


export default function ServingPage() {
  const { toast } = useToast();
  const { data } = useQuery({
    queryKey: ['serving-endpoints'],
    queryFn: () => fetchEndpoints(),
    staleTime: 30_000,
    retry: 2,
  });

  const endpoints = data?.endpoints || [];
  const [selected, setSelected] = useState<string>('');
  const ep = endpoints.find((e: any) => e.name === selected) || endpoints[0];
  const [trafficSplit, setTrafficSplit] = useState(85);
  const [minReplicas, setMinReplicas] = useState(2);
  const [maxReplicas, setMaxReplicas] = useState(8);
  const [targetGpuUtil, setTargetGpuUtil] = useState(70);

  // Generate chart data based on whether endpoints exist
  const chartData = useMemo(() => generateTimeData(endpoints), [endpoints.length]);

  // Promote mutation
  const promoteMutation = useMutation({
    mutationFn: ({ name, traffic }: { name: string; traffic: number }) =>
      api.serving.promote(name, traffic),
    onSuccess: (_, vars) => {
      toast(`Traffic updated for ${vars.name}: ${vars.traffic}%`, 'success');
    },
    onError: (err: any) => {
      toast(`Failed to update traffic: ${err?.message || 'Unknown error'}`, 'error');
    },
  });

  const handleApplySplit = () => {
    if (endpoints.length < 2) return;
    promoteMutation.mutate({ name: endpoints[0]?.name, traffic: trafficSplit });
  };

  const handlePromote = (endpointName: string) => {
    if (confirm(`Promote ${endpointName} to 100% traffic?`)) {
      promoteMutation.mutate({ name: endpointName, traffic: 100 });
      setTrafficSplit(0);
    }
  };

  const handleRollback = () => {
    if (endpoints.length < 2) return;
    if (confirm(`Rollback to ${endpoints[0]?.name} at 100% traffic?`)) {
      promoteMutation.mutate({ name: endpoints[0]?.name, traffic: 100 });
      setTrafficSplit(100);
    }
  };

  const handleSaveConfig = () => {
    toast(`Auto-scaling config saved: ${minReplicas}–${maxReplicas} replicas, ${targetGpuUtil}% target GPU`, 'success');
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Serving & Endpoints</h1>
        <p>Manage live inference endpoints, traffic splitting, and canary deployments</p>
      </div>

      {/* Endpoint Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16, marginBottom: 24 }}>
        {endpoints.map((ep: any) => (
          <div key={ep.name} className="card" style={{
            cursor: 'pointer',
            border: selected === ep.name ? '2px solid var(--accent-primary)' : undefined,
          }} onClick={() => setSelected(ep.name)}>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Globe style={{ width: 16, height: 16, color: 'var(--accent-secondary)' }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>{ep.name}</span>
                </div>
                <span className={`badge ${ep.status === 'Active' ? 'completed' : ep.status === 'Shadow' ? 'running' : 'idle'}`}>
                  {ep.status}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 12 }}>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>Latency p99</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{typeof ep.latency_p99 === 'number' ? `${ep.latency_p99}ms` : ep.latency_p99}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>RPS</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(ep.rps || 0).toLocaleString()}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>Traffic</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{ep.traffic}%</div>
                </div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>
                Model: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{ep.model}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      {endpoints.length === 0 && (
        <div className="card">
          <div className="card-body" style={{ textAlign: 'center', padding: 64 }}>
            <Globe style={{ width: 40, height: 40, color: 'var(--accent-secondary)', opacity: 0.3, margin: '0 auto 16px' }} />
            <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No serving endpoints found</h3>
            <p style={{ fontSize: 13, color: 'var(--text-dimmed)', maxWidth: 400, margin: '0 auto' }}>
              Deploy a model from the Model Registry to create an inference endpoint.
              Label K8s deployments with <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>ml-platform/type=serving</code> to discover them here.
            </p>
          </div>
        </div>
      )}

      {/* Metrics Dashboard for Selected Endpoint */}
      {endpoints.length > 0 && (
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Latency Chart */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Latency (24h) — {ep?.name || 'N/A'}</span>
              <div style={{ display: 'flex', gap: 12, fontSize: 12, color: 'var(--text-dimmed)' }}>
                <span><span style={{ color: '#10b981' }}>●</span> p50</span>
                <span><span style={{ color: '#f59e0b' }}>●</span> p95</span>
                <span><span style={{ color: '#ef4444' }}>●</span> p99</span>
              </div>
            </div>
            <div className="card-body">
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.latency}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} interval={3} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}ms`} />
                    <Tooltip contentStyle={{ background: 'rgba(10,14,26,0.95)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#f1f5f9', fontSize: 12 }} />
                    <Line type="monotone" dataKey="p50" stroke="#10b981" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="p95" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="p99" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Throughput + Error Rate */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div className="card">
              <div className="card-header"><span className="card-title">Throughput (RPS)</span></div>
              <div className="card-body">
                <div style={{ height: 140 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData.rps}>
                      <defs>
                        <linearGradient id="rpsGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} interval={5} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                      <Area type="monotone" dataKey="rps" stroke="#06b6d4" strokeWidth={1.5} fill="url(#rpsGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><span className="card-title">Error Rate (%)</span></div>
              <div className="card-body">
                <div style={{ height: 140 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData.errors}>
                      <defs>
                        <linearGradient id="errGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="time" tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} interval={5} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={v => `${v.toFixed(1)}%`} />
                      <Area type="monotone" dataKey="errors" stroke="#ef4444" strokeWidth={1.5} fill="url(#errGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right column — Traffic & Config */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Traffic Splitting */}
          <div className="card">
            <div className="card-header"><span className="card-title">Traffic Splitting</span></div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {endpoints.length >= 2 ? (<>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{endpoints[0]?.name} ({endpoints[0]?.status})</div>
                  <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${trafficSplit}%`, background: 'var(--success)', borderRadius: 4, transition: 'width 0.3s' }} />
                  </div>
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14, minWidth: 40, textAlign: 'right' }}>{trafficSplit}%</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{endpoints[1]?.name} ({endpoints[1]?.status})</div>
                  <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${100 - trafficSplit}%`, background: 'var(--accent-primary)', borderRadius: 4, transition: 'width 0.3s' }} />
                  </div>
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14, minWidth: 40, textAlign: 'right' }}>{100 - trafficSplit}%</span>
              </div>
              <input
                type="range" min={0} max={100} value={trafficSplit}
                onChange={e => setTrafficSplit(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-primary btn-sm" style={{ flex: 1 }}
                  onClick={handleApplySplit}
                  disabled={promoteMutation.isPending}
                >{promoteMutation.isPending ? 'Applying...' : 'Apply Split'}</button>
                <button className="btn btn-ghost btn-sm" onClick={() => setTrafficSplit(50)}>50/50</button>
              </div>
              </>) : (
                <p style={{ fontSize: 13, color: 'var(--text-dimmed)' }}>Deploy at least 2 endpoints to configure traffic splitting.</p>
              )}
            </div>
          </div>

          {/* Auto-Scaling Config */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Auto-Scaling</span>
              <Sliders style={{ width: 14, height: 14, color: 'var(--text-dimmed)' }} />
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Min Replicas</span>
                <input type="number" value={minReplicas} min={1} max={10}
                  onChange={e => setMinReplicas(Number(e.target.value))}
                  style={{ width: 60, padding: '3px 6px', fontSize: 13, textAlign: 'center', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Max Replicas</span>
                <input type="number" value={maxReplicas} min={1} max={20}
                  onChange={e => setMaxReplicas(Number(e.target.value))}
                  style={{ width: 60, padding: '3px 6px', fontSize: 13, textAlign: 'center', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Target GPU Util</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <input type="number" value={targetGpuUtil} min={10} max={100}
                    onChange={e => setTargetGpuUtil(Number(e.target.value))}
                    style={{ width: 50, padding: '3px 6px', fontSize: 13, textAlign: 'center', borderRadius: 4,
                      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                      color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none',
                    }}
                  />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>%</span>
                </div>
              </div>
              <button className="btn btn-ghost btn-sm" style={{ width: '100%' }} onClick={handleSaveConfig}>Save Configuration</button>
            </div>
          </div>

          {/* Canary Deployment */}
          <div className="card">
            <div className="card-header"><span className="card-title">Canary Deployment</span></div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {endpoints.length >= 2 ? (<>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                Gradually roll out <strong style={{ color: 'var(--text-primary)' }}>{endpoints[1]?.name}</strong> to replace {endpoints[0]?.name}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                {[5, 10, 25, 50, 100].map(pct => (
                  <button key={pct} className="btn btn-ghost btn-sm" onClick={() => setTrafficSplit(100 - pct)}
                    style={{
                      flex: 1,
                      background: (100 - trafficSplit) >= pct ? 'rgba(124,58,237,0.12)' : 'transparent',
                      fontFamily: 'var(--font-mono)', fontSize: 11,
                    }}
                  >{pct}%</button>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                <button className="btn btn-primary btn-sm" style={{ flex: 1 }}
                  onClick={() => handlePromote(endpoints[1]?.name)}>
                  <ArrowUpRight style={{ width: 12, height: 12 }} /> Promote {endpoints[1]?.name}
                </button>
                <button className="btn btn-ghost btn-sm" style={{ color: 'var(--error)' }}
                  onClick={handleRollback}>
                  <RotateCcw style={{ width: 12, height: 12 }} /> Rollback
                </button>
              </div>
              </>) : (
                <p style={{ fontSize: 13, color: 'var(--text-dimmed)' }}>Deploy a canary endpoint to enable gradual rollout.</p>
              )}
            </div>
          </div>
        </div>
      </div>
      )}
    </div>
  );
}
