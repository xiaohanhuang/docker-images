
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Plus, Square, X, Monitor, BookOpen, Waves, Loader2 } from 'lucide-react';
import { fetchDesks, api } from '@/lib/api';

/* ──────────── IDE tool tabs ──────────── */
const IDE_TABS = ['vscode', 'jupyter', 'marimo'] as const;

/* ──────────── New Desk Modal ──────────── */
function NewDeskModal({ open, onClose, onLaunch, isLaunching }: {
  open: boolean;
  onClose: () => void;
  onLaunch: (spec: { name: string; image: string; gpu_type: string; gpu_count: number; cpu_count: number; memory?: string }) => void;
  isLaunching: boolean;
}) {
  const [name, setName] = useState('');
  const [image, setImage] = useState('ml-platform/desk-gpu:1.0.0');
  const [gpuType, setGpuType] = useState('CPU');
  const [gpuCount, setGpuCount] = useState(0.25); // Set default to 1/4 GPU
  const [cpuCount, setCpuCount] = useState(4);
  const [memory, setMemory] = useState(16); // 16 GB default

  if (!open) return null;

  // Real AWS on-demand pricing (us-west-2, March 2026)
  // rate = per-GPU cost (instance price / GPUs per instance)
  // validGpus = requested physical GPUs (e.g. 0.25 means 1 time-sliced replica)
  const gpuOptions: { type: string; instance: string; ratePerGpu: number; validGpus: number[] }[] = [
    { type: 'CPU',  instance: 'm5',    ratePerGpu: 0.192 / 4,  validGpus: [] },
    { type: 'T4',   instance: 'g4dn',  ratePerGpu: 0.526,  validGpus: [0.25, 0.5, 1, 2, 4, 8] },
    { type: 'A10G', instance: 'g5',    ratePerGpu: 1.006,  validGpus: [0.25, 0.5, 1, 2, 4, 8] },
    { type: 'A100', instance: 'p4d',   ratePerGpu: 4.10,   validGpus: [0.25, 0.5, 1, 2, 4, 8] },
    { type: 'H100', instance: 'p5',    ratePerGpu: 6.88,   validGpus: [0.25, 0.5, 1, 2, 4, 8] },
  ];
  const images = ['ml-platform/desk-gpu:1.0.0', 'ml-platform/base-cpu:1.1.0', 'ml-platform/desk-cpu:1.0.0', 'ml-platform/base-gpu:1.1.0', 'ml-platform/training-llm:1.1.0'];
  const selected = gpuOptions.find(o => o.type === gpuType) || gpuOptions[0];
  const hourly = selected.validGpus.length === 0 ? selected.ratePerGpu * cpuCount : selected.ratePerGpu * gpuCount;
  const isGpu = selected.validGpus.length > 0;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 1000, backdropFilter: 'blur(4px)',
    }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        width: 500, borderRadius: 'var(--radius-lg)',
        background: 'rgba(15,20,35,0.98)', border: '1px solid rgba(255,255,255,0.1)',
        boxShadow: '0 24px 48px rgba(0,0,0,0.5)',
      }}>
        {/* Header */}
        <div style={{ padding: '20px 24px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3 style={{ fontSize: 16, fontWeight: 700 }}>Launch New Desk</h3>
            <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dimmed)' }}>
              <X style={{ width: 16, height: 16 }} />
            </button>
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-dimmed)', marginTop: 4 }}>
            Create a new interactive development environment
          </p>
        </div>

        {/* Form */}
        <div style={{ padding: '16px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Name */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
              Desk Name
            </label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="desk-my-project"
              style={{
                width: '100%', padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)',
                color: 'var(--text-primary)', fontSize: 13, outline: 'none',
                fontFamily: 'var(--font-mono)',
              }}
            />
          </div>

          {/* Image */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
              Docker Image
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {images.map(img => (
                <button key={img} onClick={() => setImage(img)}
                  className={`btn ${image === img ? 'btn-primary' : 'btn-ghost'} btn-sm`}
                  style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                  {img}
                </button>
              ))}
            </div>
          </div>

          {/* Hardware Type */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
              Hardware
            </label>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {gpuOptions.map(opt => (
                <button key={opt.type} onClick={() => { setGpuType(opt.type); setGpuCount(1); }}
                  className={`btn ${gpuType === opt.type ? 'btn-primary' : 'btn-ghost'} btn-sm`}
                  style={{ flexDirection: 'column', alignItems: 'center', padding: '6px 12px', lineHeight: 1.3 }}>
                  <span>{opt.type}</span>
                  <span style={{ fontSize: 10, opacity: 0.6 }}>
                    {opt.validGpus.length > 0 ? `$${opt.ratePerGpu.toFixed(2)}/gpu/hr` : `$${opt.ratePerGpu.toFixed(2)}/cpu/hr`}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* GPU Count — shown for all GPU types */}
          {isGpu && selected.validGpus.length > 1 && (
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                GPU Count: {gpuCount < 1 ? `1/${Math.round(1/gpuCount)}` : gpuCount}
              </label>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {selected.validGpus.map(n => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setGpuCount(n)}
                    className={`btn ${gpuCount === n ? 'btn-primary' : ''}`}
                    style={{ flex: 1, padding: '4px 8px', fontSize: 11, background: gpuCount === n ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)' }}
                  >
                    {n < 1 ? `1/${Math.round(1/n)}` : n}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* CPU & Memory config for CPU machines */}
          {!isGpu && (
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                  CPU Count: {cpuCount}
                </label>
                <input type="range" min="1" max="256" value={cpuCount}
                  onChange={e => setCpuCount(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dimmed)', marginTop: 2 }}>
                  <span>1</span><span>128</span><span>256</span>
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                  Memory: {memory < 1 ? '500 MB' : `${memory} GB`}
                </label>
                <input type="range" min="0.5" max="256" step="0.5" value={memory}
                  onChange={e => setMemory(Number(e.target.value))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dimmed)', marginTop: 2 }}>
                  <span>0.5G</span><span>128G</span><span>256G</span>
                </div>
              </div>
            </div>
          )}

          {/* Cost estimate */}
          <div style={{
            fontSize: 13, display: 'flex', flexDirection: 'column', gap: 6,
            padding: '12px 16px', background: 'rgba(255,255,255,0.03)', borderRadius: 8,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Estimated cost</span>
              <span style={{ fontWeight: 600, color: 'var(--cost-green)' }}>${hourly.toFixed(2)}/hr</span>
            </div>
            {isGpu ? (
              <span style={{ fontSize: 11, color: 'var(--text-dimmed)' }}>
                {gpuCount < 1 ? `1/${Math.round(1/gpuCount)}` : gpuCount}× {gpuType} @ ${selected.ratePerGpu.toFixed(2)}/gpu/hr ({selected.instance} family)
              </span>
            ) : (
              <span style={{ fontSize: 11, color: 'var(--text-dimmed)' }}>
                {cpuCount} cpus @ ${selected.ratePerGpu.toFixed(2)}/cpu/hr ({selected.instance} family)
              </span>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button onClick={onClose} className="btn btn-ghost">Cancel</button>
          <button
            onClick={() => onLaunch({
              name: name || `desk-${Date.now().toString(36)}`,
              image,
              gpu_type: gpuType,
              gpu_count: gpuType === 'CPU' ? 0 : Math.round(gpuCount * 4), // 4 slices per physical GPU
              cpu_count: cpuCount,
              memory: memory === 0.5 ? '500Mi' : `${memory}Gi`,
            })}
            className="btn btn-primary"
            disabled={isLaunching}
          >
            {isLaunching ? (
              <><Loader2 style={{ width: 14, height: 14, animation: 'spin 1s linear infinite' }} /> Launching...</>
            ) : (
              <><Plus style={{ width: 14, height: 14 }} /> Launch Desk</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ──────────── Status Component ──────────── */
function StatusDot({ status }: { status: string }) {
  const s = status.toLowerCase();
  const cls = s === 'running' ? 'running' : s === 'idle' ? 'idle'
    : s === 'stopped' || s === 'terminated' ? 'failed' : 'completed';
  return (
    <span className={`badge ${cls}`}>
      <span className="badge-dot" />
      {status}
    </span>
  );
}

/* ──────────── Main Page ──────────── */
export default function DesksPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showNewModal, setShowNewModal] = useState(false);
  const [stoppingId, setStoppingId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ['desks'],
    queryFn: () => fetchDesks(),
    staleTime: 30_000,
    retry: 2,
  });

  const desks = data?.desks || [];

  // Launch mutation
  const launchMutation = useMutation({
    mutationFn: (spec: { name: string; image: string; gpu_type: string; gpu_count: number; cpu_count: number }) =>
      api.desks.launch(spec),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['desks'] });
      setShowNewModal(false);
    },
    onError: (err: any) => {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error';
      alert(`Failed to launch desk: ${msg}`);
    },
  });

  // Stop mutation
  const stopMutation = useMutation({
    mutationFn: (deskId: string) => api.desks.stop(deskId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['desks'] });
      setStoppingId(null);
    },
    onError: () => setStoppingId(null),
  });

  const handleStop = (e: React.MouseEvent, deskId: string) => {
    e.stopPropagation();
    if (confirm(`Stop desk ${deskId}? Unsaved in-memory state will be lost.`)) {
      setStoppingId(deskId);
      stopMutation.mutate(deskId);
    }
  };

  const handleOpenTool = (e: React.MouseEvent, desk: any, tab: string) => {
    e.stopPropagation();
    navigate(`/desks/${desk.id}?tab=${tab}`);
  };

  return (
    <div className="page-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>My Desks</h1>
          <p>Manage your development environments</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowNewModal(true)}>
          <Plus style={{ width: 16, height: 16 }} />
          New Desk
        </button>
      </div>

      {/* Desk Grid */}
      {desks.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 20 }}>
          {desks.map((c: any) => (
            <div key={c.id} className="card" style={{ cursor: 'pointer' }} onClick={() => navigate(`/desks/${c.id}`)}>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <Box style={{ width: 18, height: 18, color: 'var(--accent-primary)' }} />
                    <span style={{ fontWeight: 600, fontSize: 15 }}>{c.name || c.id}</span>
                  </div>
                  <StatusDot status={c.status} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>GPU</div>
                    <div style={{ fontSize: 14, fontFamily: 'var(--font-mono)' }}>{c.gpu}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Uptime</div>
                    <div style={{ fontSize: 14 }}>{c.uptime}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Burn Rate</div>
                    <div style={{ fontSize: 14, color: 'var(--cost-green)' }}>{c.burn_rate}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Image</div>
                    <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.image}</div>
                  </div>
                </div>
                {(c.status === 'Running' || c.status === 'Pending') && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    {c.status === 'Running' && (
                      <>
                        <button className="btn btn-ghost btn-sm" onClick={(e) => handleOpenTool(e, c, 'vscode')}>
                          <Monitor style={{ width: 12, height: 12 }} /> VS Code
                        </button>
                        <button className="btn btn-ghost btn-sm" onClick={(e) => handleOpenTool(e, c, 'jupyter')}>
                          <BookOpen style={{ width: 12, height: 12 }} /> Notebook
                        </button>
                        <button className="btn btn-ghost btn-sm" onClick={(e) => handleOpenTool(e, c, 'marimo')}>
                          <Waves style={{ width: 12, height: 12 }} /> Marimo
                        </button>
                      </>
                    )}
                    <button className="btn btn-ghost btn-sm" style={{ color: 'var(--error)', marginLeft: 'auto' }}
                      onClick={(e) => handleStop(e, c.id)}
                      disabled={stoppingId === c.id}
                    >
                      {stoppingId === c.id ? (
                        <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
                      ) : (
                        <Square style={{ width: 12, height: 12 }} />
                      )}
                      {stoppingId === c.id ? 'Stopping...' : 'Stop'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Empty State */
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          padding: '80px 40px', borderRadius: 'var(--radius-lg)',
          border: '1px dashed rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.02)',
        }}>
          <Box style={{ width: 48, height: 48, color: 'var(--text-dimmed)', marginBottom: 16 }} />
          <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>No desks yet</h3>
          <p style={{ color: 'var(--text-dimmed)', fontSize: 14, marginBottom: 24, textAlign: 'center', maxWidth: 400 }}>
            Launch a desk to get a GPU-powered development environment with VS Code, Jupyter, and a terminal — all in your browser.
          </p>
          <button className="btn btn-primary" onClick={() => setShowNewModal(true)}>
            <Plus style={{ width: 16, height: 16 }} /> Launch Your First Desk
          </button>
        </div>
      )}

      {/* New Desk Modal */}
      <NewDeskModal
        open={showNewModal}
        onClose={() => setShowNewModal(false)}
        onLaunch={(spec) => launchMutation.mutate(spec)}
        isLaunching={launchMutation.isPending}
      />
    </div>
  );
}
