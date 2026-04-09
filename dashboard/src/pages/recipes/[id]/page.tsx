
import { useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft, Play, Zap, ChevronRight, Settings, Box,
  Cpu, DollarSign, Clock, AlertTriangle, Shield, Loader2,
} from 'lucide-react';
import { apiClient } from '@/lib/api';

export default function RecipeLauncherPage() {
  const params = useParams();
  const navigate = useNavigate();
  const recipeId = params.id as string;

  const { data: recipe, isLoading } = useQuery({
    queryKey: ['recipe-detail', recipeId],
    queryFn: async () => {
      const { data } = await apiClient.get(`/recipes/${encodeURIComponent(recipeId)}`);
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });

  const r = recipe || {
    name: recipeId, version: '?', verified: false,
    description: 'Loading recipe details...',
    author: '—', tags: [],
    steps: [], profiles: [{ name: 'Default', gpu: 'T4 x1', cost: '—', desc: 'Default profile', ram: '—', vram: '—' }],
    params: [],
  };

  const [selectedProfile, setSelectedProfile] = useState(0);
  const [paramValues, setParamValues] = useState<Record<string, any>>({});
  const [computeType, setComputeType] = useState<'on-demand' | 'spot'>('on-demand');
  const [requirements, setRequirements] = useState('');
  const [isLaunching, setIsLaunching] = useState(false);

  // Initialize param defaults when recipe loads
  if (recipe && Object.keys(paramValues).length === 0 && r.params.length > 0) {
    const defaults: Record<string, any> = {};
    r.params.forEach((p: any) => { defaults[p.key] = p.default; });
    setParamValues(defaults);
  }

  const handleLaunch = (canary = false) => {
    setIsLaunching(true);
    setTimeout(() => {
      setIsLaunching(false);
      navigate('/jobs');
    }, 2000);
  };

  if (isLoading) {
    return (
      <div className="page-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        <Loader2 style={{ width: 24, height: 24, animation: 'spin 1s linear infinite', color: 'var(--accent-primary)' }} />
        <span style={{ marginLeft: 12, color: 'var(--text-muted)' }}>Loading recipe {recipeId}...</span>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Back navigation */}
      <div style={{ marginBottom: 16 }}>
        <button
          onClick={() => navigate('/recipes')}
          style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'none', border: 'none', color: 'var(--text-dimmed)', cursor: 'pointer', fontSize: 13 }}
        >
          <ArrowLeft style={{ width: 14, height: 14 }} /> Recipe Catalog
        </button>
      </div>

      {/* Recipe Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
            <h1 style={{ margin: 0 }}>{r.name}</h1>
            <span style={{ fontSize: 14, color: 'var(--text-dimmed)', fontFamily: 'var(--font-mono)' }}>{r.version}</span>
            {r.verified && (
              <span className="badge completed" style={{ fontSize: 11 }}>
                <Shield style={{ width: 10, height: 10 }} /> VERIFIED
              </span>
            )}
          </div>
          <p style={{ margin: 0 }}>{r.description}</p>
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            {r.tags.map((tag: string) => (
              <span key={tag} style={{
                fontSize: 11, fontWeight: 500, color: 'var(--accent-secondary)',
                background: 'rgba(6,182,212,0.1)', padding: '2px 8px', borderRadius: 100,
              }}>{tag}</span>
            ))}
            <span style={{ fontSize: 12, color: 'var(--text-dimmed)', marginLeft: 8 }}>by {r.author}</span>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24, marginTop: 8 }}>
        {/* Left Column — Pipeline + Params */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Pipeline Steps Timeline */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Pipeline Steps</span>
              <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{r.steps.length} stages</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
              {r.steps.map((step: any, i: number) => (
                <div key={i}>
                  {i > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', paddingLeft: 16, height: 20 }}>
                      <div style={{ width: 2, height: 20, background: 'var(--accent-primary)', opacity: 0.4 }} />
                    </div>
                  )}
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
                    background: 'rgba(124,58,237,0.04)',
                    border: '1px solid rgba(124,58,237,0.12)',
                    borderRadius: 'var(--radius-sm)',
                  }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: '50%',
                      background: 'rgba(124,58,237,0.2)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 12, fontWeight: 700, color: 'var(--accent-primary)',
                    }}>{i + 1}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{step.name}</div>
                      <div style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{step.description}</div>
                    </div>
                    <ChevronRight style={{ width: 14, height: 14, color: 'var(--text-dimmed)' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Parameters Form */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Parameters</span>
              <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{r.params.length} configurable</span>
            </div>
            <div className="card-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                {r.params.map((param: any) => (
                  <div key={param.key}>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>
                      {param.label}
                    </label>
                    {param.type === 'select' ? (
                      <select
                        value={paramValues[param.key]}
                        onChange={e => setParamValues({ ...paramValues, [param.key]: e.target.value })}
                        style={{
                          width: '100%', padding: '8px 12px', fontSize: 13,
                          borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.1)',
                          background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)',
                          fontFamily: 'var(--font-mono)', outline: 'none',
                        }}
                      >
                        {param.options.map((opt: string) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={param.type === 'number' ? 'number' : 'text'}
                        value={paramValues[param.key]}
                        onChange={e => setParamValues({ ...paramValues, [param.key]: param.type === 'number' ? Number(e.target.value) : e.target.value })}
                        style={{
                          width: '100%', padding: '8px 12px', fontSize: 13,
                          borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.1)',
                          background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)',
                          fontFamily: 'var(--font-mono)', outline: 'none',
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Environment & Dependencies */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Environment & Dependencies</span>
            </div>
            <div className="card-body">
              <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>
                Additional pip requirements (one per line)
              </label>
              <textarea
                value={requirements}
                onChange={e => setRequirements(e.target.value)}
                placeholder="transformers>=4.38.0&#10;datasets&#10;peft>=0.8.0"
                rows={4}
                style={{
                  width: '100%', padding: '10px 14px', fontSize: 13,
                  borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)',
                  fontFamily: 'var(--font-mono)', outline: 'none', resize: 'vertical',
                }}
              />
              <p style={{ fontSize: 11, color: 'var(--text-dimmed)', marginTop: 6 }}>
                Dependencies are built and cached via an Init Container before the GPU container starts.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column — Infrastructure Profile + Launch */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Infrastructure Profile Selector */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Infrastructure Profile</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {r.profiles.map((profile: any, i: number) => (
                <div
                  key={profile.name}
                  onClick={() => setSelectedProfile(i)}
                  style={{
                    padding: '14px 16px', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                    border: selectedProfile === i
                      ? '2px solid var(--accent-primary)'
                      : '1px solid rgba(255,255,255,0.08)',
                    background: selectedProfile === i
                      ? 'rgba(124,58,237,0.08)'
                      : 'rgba(255,255,255,0.02)',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontWeight: 600, fontSize: 15 }}>{profile.name}</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700,
                      color: 'var(--cost-green)',
                    }}>{profile.cost}</span>
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>{profile.desc}</div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-dimmed)' }}>
                      <Cpu style={{ width: 12, height: 12 }} /> {profile.gpu}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-dimmed)' }}>
                      <Box style={{ width: 12, height: 12 }} /> {profile.ram} RAM
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-dimmed)' }}>
                      <Zap style={{ width: 12, height: 12 }} /> {profile.vram} VRAM
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Compute Lifecycle */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Compute Lifecycle</span>
            </div>
            <div className="card-body">
              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  onClick={() => setComputeType('on-demand')}
                  style={{
                    flex: 1, padding: '10px 14px', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                    border: computeType === 'on-demand' ? '2px solid var(--accent-primary)' : '1px solid rgba(255,255,255,0.08)',
                    background: computeType === 'on-demand' ? 'rgba(124,58,237,0.08)' : 'rgba(255,255,255,0.02)',
                    color: 'var(--text-primary)', fontSize: 13, fontWeight: 500, textAlign: 'center',
                  }}
                >
                  On-Demand
                </button>
                <button
                  onClick={() => setComputeType('spot')}
                  style={{
                    flex: 1, padding: '10px 14px', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                    border: computeType === 'spot' ? '2px solid var(--warning)' : '1px solid rgba(255,255,255,0.08)',
                    background: computeType === 'spot' ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.02)',
                    color: 'var(--text-primary)', fontSize: 13, fontWeight: 500, textAlign: 'center',
                  }}
                >
                  Spot
                </button>
              </div>
              {computeType === 'spot' && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 6, marginTop: 10,
                  padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                  background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
                  fontSize: 12, color: 'var(--warning)',
                }}>
                  <AlertTriangle style={{ width: 14, height: 14, flexShrink: 0 }} />
                  May be interrupted. Ensure checkpointing is configured.
                </div>
              )}
            </div>
          </div>

          {/* Cost Summary */}
          <div className="card" style={{ background: 'rgba(16,185,129,0.04)', border: '1px solid rgba(16,185,129,0.15)' }}>
            <div className="card-body">
              <div style={{ fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-dimmed)', marginBottom: 10 }}>
                Cost Estimate
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Profile</span>
                <span style={{ fontWeight: 500 }}>{r.profiles[selectedProfile]?.name} ({r.profiles[selectedProfile]?.gpu})</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Compute</span>
                <span style={{ fontWeight: 500 }}>{computeType === 'spot' ? 'Spot (~60% savings)' : 'On-Demand'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-muted)' }}>Hourly Rate</span>
                <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--cost-green)' }}>
                  {computeType === 'spot'
                    ? `~$${(parseFloat(recipe.profiles[selectedProfile]?.cost?.replace(/[^0-9.]/g, '') || '0') * 0.4).toFixed(0)}/hr`
                    : recipe.profiles[selectedProfile]?.cost
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Launch Buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <button
              className="btn btn-primary"
              onClick={() => handleLaunch(false)}
              disabled={isLaunching}
              style={{
                width: '100%', padding: '14px', fontSize: 15, fontWeight: 600,
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              }}
            >
              {isLaunching ? (
                'Launching...'
              ) : (
                <>
                  <Play style={{ width: 16, height: 16 }} fill="currentColor" /> Launch Recipe
                </>
              )}
            </button>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => handleLaunch(false)}>
                <Settings style={{ width: 14, height: 14 }} /> Dry Run
              </button>
              <button className="btn btn-ghost" style={{ flex: 1 }} onClick={() => handleLaunch(true)}>
                <Zap style={{ width: 14, height: 14 }} /> Canary Probe
              </button>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-dimmed)', textAlign: 'center' }}>
              Canary runs with <code style={{ background: 'rgba(255,255,255,0.06)', padding: '1px 4px', borderRadius: 3 }}>ML_PLAT_IS_CANARY=True</code> and a 5-minute hard timeout.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
