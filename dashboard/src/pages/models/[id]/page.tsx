
import { useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft, ArrowUpRight, CheckCircle, Clock, Database,
  GitPullRequest, Shield, Tag, AlertTriangle, Eye, Download,
  ChevronRight, FileText, Activity, Loader2,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api, apiClient } from '@/lib/api';

export default function ModelDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  const modelId = params.id as string;

  // Fetch real model data from MLflow
  const { data: modelData, isLoading, error } = useQuery({
    queryKey: ['model-detail', modelId],
    queryFn: async () => {
      const { data } = await apiClient.get(`/mlflow/models/${encodeURIComponent(modelId)}`);
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });

  const [promotionOpen, setPromotionOpen] = useState(false);
  const [promoteVersion, setPromoteVersion] = useState('');
  const [promoteTarget, setPromoteTarget] = useState('Staging');
  const [approver, setApprover] = useState('');

  const stageColors: Record<string, string> = {
    Production: 'var(--success)',
    Staging: '#f59e0b',
    Archived: '#64748b',
    None: '#64748b',
  };

  // Parse MLflow model data into display format
  const rm = modelData?.registered_model || {};
  const model = {
    name: rm.name || modelId,
    description: rm.description || 'Model registered in MLflow',
    tags: (rm.tags || []).map((t: any) => t.key || t),
    versions: (rm.latest_versions || []).map((v: any) => ({
      version: `v${v.version}`,
      stage: v.current_stage || 'None',
      metric: '',
      created: v.creation_timestamp ? new Date(parseInt(v.creation_timestamp)).toLocaleDateString() : '—',
      size: '—',
      format: v.source?.includes('safetensors') ? 'SafeTensors' : 'PyTorch',
      commit: v.run_id?.substring(0, 7) || '—',
      run: v.run_id || '',
    })),
    metrics_history: [] as any[],
    serving: [] as any[],
    created: rm.creation_timestamp ? new Date(parseInt(rm.creation_timestamp)).toLocaleDateString() : '—',
    updated: rm.last_updated_timestamp ? new Date(parseInt(rm.last_updated_timestamp)).toLocaleDateString() : '—',
    author: '—',
  };

  if (isLoading) {
    return (
      <div className="page-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        <Loader2 style={{ width: 24, height: 24, animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />
        <span style={{ marginLeft: 12, color: 'var(--text-muted)' }}>Loading model {modelId}...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <button onClick={() => navigate('/models')}
          style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', color: 'var(--text-dimmed)', cursor: 'pointer', fontSize: 13, marginBottom: 16 }}>
          <ArrowLeft style={{ width: 14, height: 14 }} /> Model Registry
        </button>
        <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-dimmed)' }}>
          <Database style={{ width: 32, height: 32, margin: '0 auto 12px', opacity: 0.4 }} />
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>Model not found</div>
          <p style={{ fontSize: 13 }}>&ldquo;{modelId}&rdquo; was not found in the MLflow model registry.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Back navigation */}
      <button onClick={() => navigate('/models')}
        style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', color: 'var(--text-dimmed)', cursor: 'pointer', fontSize: 13, marginBottom: 16 }}>
        <ArrowLeft style={{ width: 14, height: 14 }} /> Model Registry
      </button>

      {/* Model Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
            <Database style={{ width: 24, height: 24, color: 'var(--accent-primary)' }} />
            <h1 style={{ margin: 0 }}>{model.name}</h1>
          </div>
          <p style={{ margin: 0, marginBottom: 8 }}>{model.description}</p>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {model.tags.map((tag: string) => (
              <span key={tag} style={{ fontSize: 11, fontWeight: 500, color: 'var(--accent-secondary)', background: 'rgba(6,182,212,0.1)', padding: '2px 8px', borderRadius: 100 }}>{tag}</span>
            ))}
            <span style={{ fontSize: 12, color: 'var(--text-dimmed)', marginLeft: 8 }}>Created {model.created}</span>
          </div>
        </div>
        <button className="btn btn-primary" onClick={() => { setPromotionOpen(true); setPromoteVersion(model.versions[0]?.version || ''); }}>
          <GitPullRequest style={{ width: 14, height: 14 }} /> Request Promotion
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24, marginTop: 8 }}>
        {/* Left Column — Versions */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Version History */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Version History</span>
              <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{model.versions.length} versions</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {model.versions.length > 0 ? model.versions.map((v: any) => (
                <div key={v.version} style={{
                  display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px',
                  borderRadius: 'var(--radius-sm)',
                  border: v.stage === 'Production' ? '1px solid rgba(16,185,129,0.2)' : '1px solid rgba(255,255,255,0.06)',
                  background: v.stage === 'Production' ? 'rgba(16,185,129,0.04)' : 'rgba(255,255,255,0.02)',
                }}>
                  <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-mono)', minWidth: 36 }}>{v.version}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span className={`badge ${v.stage === 'Production' ? 'completed' : v.stage === 'Staging' ? 'running' : 'idle'}`}>
                        {v.stage}
                      </span>
                      {v.metric && <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{v.metric}</span>}
                    </div>
                    <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-dimmed)' }}>
                      <span>{v.created}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{v.commit}</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {v.stage !== 'Production' && v.stage !== 'Archived' && (
                      <button className="btn btn-ghost btn-sm" onClick={() => { setPromoteVersion(v.version); setPromotionOpen(true); }}>
                        <ArrowUpRight style={{ width: 12, height: 12 }} /> Promote
                      </button>
                    )}
                  </div>
                </div>
              )) : (
                <div style={{ fontSize: 13, color: 'var(--text-dimmed)', textAlign: 'center', padding: 20 }}>
                  No versions registered yet
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column — Info + Promotion */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Model Card Info */}
          <div className="card">
            <div className="card-header"><span className="card-title">Model Info</span></div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dimmed)' }}>Name</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{model.name}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dimmed)' }}>Versions</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{model.versions.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dimmed)' }}>Created</span>
                <span>{model.created}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dimmed)' }}>Updated</span>
                <span>{model.updated}</span>
              </div>
            </div>
          </div>

          {/* Promotion PR Modal */}
          {promotionOpen && (
            <div className="card" style={{ border: '1px solid rgba(124,58,237,0.3)', background: 'rgba(124,58,237,0.04)' }}>
              <div className="card-header">
                <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <GitPullRequest style={{ width: 16, height: 16 }} /> Promotion Request
                </span>
                <button className="btn btn-ghost btn-sm" onClick={() => setPromotionOpen(false)} style={{ fontSize: 18, lineHeight: 1 }}>×</button>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-dimmed)', display: 'block', marginBottom: 4 }}>Version</label>
                  <select value={promoteVersion} onChange={e => setPromoteVersion(e.target.value)}
                    style={{
                      width: '100%', padding: '8px 12px', fontSize: 13, borderRadius: 'var(--radius-sm)',
                      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                      color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none',
                    }}
                  >
                    {model.versions.map((v: any) => <option key={v.version} value={v.version}>{v.version} ({v.stage})</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-dimmed)', display: 'block', marginBottom: 4 }}>Target Stage</label>
                  <select value={promoteTarget} onChange={e => setPromoteTarget(e.target.value)}
                    style={{
                      width: '100%', padding: '8px 12px', fontSize: 13, borderRadius: 'var(--radius-sm)',
                      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                      color: 'var(--text-primary)', outline: 'none',
                    }}
                  >
                    <option value="Staging">Staging</option>
                    <option value="Production">Production</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-dimmed)', display: 'block', marginBottom: 4 }}>Approver</label>
                  <input type="text" value={approver} onChange={e => setApprover(e.target.value)} placeholder="@team-lead"
                    style={{
                      width: '100%', padding: '8px 12px', fontSize: 13, borderRadius: 'var(--radius-sm)',
                      background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                      color: 'var(--text-primary)', outline: 'none',
                    }}
                  />
                </div>
                {promoteTarget === 'Production' && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 6, padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                    background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
                    fontSize: 12, color: 'var(--warning)',
                  }}>
                    <AlertTriangle style={{ width: 14, height: 14, flexShrink: 0 }} />
                    Production promotion requires manager approval and automated evaluation gate.
                  </div>
                )}
                <button className="btn btn-primary" style={{ width: '100%' }}>
                  <GitPullRequest style={{ width: 14, height: 14 }} /> Submit PR
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
