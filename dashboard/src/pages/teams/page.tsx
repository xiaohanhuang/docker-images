
import { Users, Settings2, Shield, Cpu } from 'lucide-react';
import { useToast } from '@/components/Toast';

export default function TeamsPage() {
  const { toast } = useToast();

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Team Spaces</h1>
        <p>Team dashboards, GPU quotas, budgets, and shared desk templates</p>
      </div>

      <div style={{
        textAlign: 'center', padding: '80px 20px',
        border: '1px dashed rgba(255,255,255,0.1)', borderRadius: 'var(--radius-lg)',
      }}>
        <Users style={{ width: 40, height: 40, margin: '0 auto 16px', color: 'var(--accent-primary)', opacity: 0.6 }} />
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>Team Management</div>
        <p style={{ fontSize: 14, color: 'var(--text-dimmed)', maxWidth: 500, margin: '0 auto', lineHeight: 1.6 }}>
          Team-based GPU quotas, budget tracking, and shared desk templates will be available
          once the team management backend is configured.
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 24 }}>
          <button className="btn btn-ghost" style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => toast('Team management requires backend configuration. See docs/teams.md for setup.', 'info')}>
            <Settings2 style={{ width: 14, height: 14 }} /> Configure Teams
          </button>
        </div>
      </div>

      {/* Preview of what teams will look like */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16, marginTop: 24, opacity: 0.4, pointerEvents: 'none' }}>
        {[
          { name: 'ML Research', members: 5, gpuQuota: '16 A100', budget: '$25,000/mo' },
          { name: 'NLP Engineering', members: 8, gpuQuota: '8 A10G', budget: '$12,000/mo' },
          { name: 'Platform Infra', members: 3, gpuQuota: '4 T4', budget: '$5,000/mo' },
        ].map(team => (
          <div key={team.name} className="card">
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Shield style={{ width: 16, height: 16, color: 'var(--accent-primary)' }} />
                <span style={{ fontWeight: 600, fontSize: 15 }}>{team.name}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 12 }}>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>Members</div>
                  <div style={{ fontWeight: 600 }}>{team.members}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>GPU Quota</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{team.gpuQuota}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-dimmed)' }}>Budget</div>
                  <div style={{ fontWeight: 600, color: 'var(--cost-green)' }}>{team.budget}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ textAlign: 'center', marginTop: 12, fontSize: 11, color: 'var(--text-dimmed)', fontStyle: 'italic' }}>
        Preview — configure backend to enable
      </div>
    </div>
  );
}
