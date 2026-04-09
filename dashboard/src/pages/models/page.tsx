
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Database, ArrowUpRight, Shield } from 'lucide-react';
import { api } from '@/lib/api';

/** Format a raw timestamp (ms number, string number, or ISO string) to relative time */
function formatRelativeTime(raw: any): string {
  if (!raw) return '—';
  if (typeof raw === 'string' && /[a-z]/i.test(raw) && raw.includes('ago')) return raw;
  const ts = typeof raw === 'string' ? (raw.includes('T') ? new Date(raw).getTime() : parseInt(raw, 10)) : raw;
  if (isNaN(ts)) return String(raw);
  const ms = ts > 1e12 ? ts : ts * 1000;
  const diff = Date.now() - ms;
  if (diff < 3600_000) return `${Math.max(1, Math.round(diff / 60_000))}m ago`;
  if (diff < 86400_000) return `${Math.round(diff / 3600_000)}h ago`;
  return `${Math.round(diff / 86400_000)}d ago`;
}

export default function ModelsPage() {
  const navigate = useNavigate();
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: () => api.mlflow.listModels(),
    staleTime: 60_000,
    retry: 2,
  });

  const modelList = models?.registered_models || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Model Registry</h1>
        <p>Model lifecycle management with governed promotion workflows</p>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Registered Models</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{modelList.length} models</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Version</th>
              <th>Stage</th>
              <th>Key Metric</th>
              <th>Updated</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {modelList.map((model: any, i: number) => {
              const stage = model.stage || model.latest_versions?.[0]?.current_stage || 'None';
              const stageClass = stage === 'Production' ? 'completed' : stage === 'Staging' ? 'running' : 'idle';
              const version = model.version || model.latest_versions?.[0]?.version;
              const updated = model.updated || model.last_updated_timestamp || model.latest_versions?.[0]?.creation_timestamp;
              return (
                <tr key={`${model.name}-${i}`} style={{ cursor: 'pointer' }} onClick={() => navigate(`/models/${model.name}`)}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Database style={{ width: 14, height: 14, color: 'var(--accent-primary)' }} />
                      <span style={{ fontWeight: 500 }}>{model.name}</span>
                    </div>
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                    {version ? `v${version}` : '—'}
                  </td>
                  <td><span className={`badge ${stageClass}`}>{stage}</span></td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-muted)' }}>
                    {model.metrics || '—'}
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>{formatRelativeTime(updated)}</td>
                  <td>
                    <button className="btn btn-ghost btn-sm" title="View Details" onClick={(e) => { e.stopPropagation(); navigate(`/models/${model.name}`); }}>
                      <ArrowUpRight style={{ width: 14, height: 14 }} />
                    </button>
                  </td>
                </tr>
              );
            })}
            {modelList.length === 0 && (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: 48 }}>
                  <Database style={{ width: 32, height: 32, color: 'var(--accent-primary)', opacity: 0.3, margin: '0 auto 12px' }} />
                  <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8, color: 'var(--text-muted)' }}>No models registered</div>
                  <div style={{ fontSize: 13, color: 'var(--text-dimmed)' }}>
                    Register a model with <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,0.06)', padding: '1px 6px', borderRadius: 3 }}>mlflow.register_model()</code> to see it here.
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
