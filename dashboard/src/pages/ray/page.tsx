
import { useQuery } from '@tanstack/react-query';
import { Activity, Cpu, Zap, HardDrive, ExternalLink } from 'lucide-react';
import { api } from '@/lib/api';

export default function RayPage() {
  const { data: clusterStatus } = useQuery({
    queryKey: ['ray-cluster'],
    queryFn: () => api.ray.getClusterStatus(),
    staleTime: 30_000,
    retry: false,
  });

  const { data: rayJobs } = useQuery({
    queryKey: ['ray-jobs'],
    queryFn: () => api.ray.getJobs(),
    staleTime: 30_000,
    retry: false,
  });

  const cluster = clusterStatus || { active_nodes: 0, total_cpus: 0, total_gpus: 0, available_cpus: 0, available_gpus: 0 };
  const jobs = rayJobs || [];

  // Compute utilization
  const cpuUtil = cluster.total_cpus > 0 ? Math.round(((cluster.total_cpus - cluster.available_cpus) / cluster.total_cpus) * 100) : 0;
  const gpuUtil = cluster.total_gpus > 0 ? Math.round(((cluster.total_gpus - cluster.available_gpus) / cluster.total_gpus) * 100) : 0;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Ray Cluster</h1>
        <p>Distributed computing cluster status and jobs</p>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Active Nodes</div>
          <div className="stat-value">{cluster.active_nodes}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total CPUs</div>
          <div className="stat-value">{cluster.total_cpus}</div>
          <div className="stat-sub"><Cpu style={{ width: 12, height: 12 }} /> {cluster.available_cpus} available ({cpuUtil}% used)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total GPUs</div>
          <div className="stat-value">{cluster.total_gpus}</div>
          <div className="stat-sub"><Zap style={{ width: 12, height: 12 }} /> {cluster.available_gpus} available ({gpuUtil}% used)</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Cluster Health</div>
          <div className="stat-value" style={{ color: cluster.active_nodes > 0 ? 'var(--success)' : 'var(--text-dimmed)' }}>
            {cluster.active_nodes > 0 ? 'Healthy' : 'No Nodes'}
          </div>
        </div>
      </div>

      {/* Resource Utilization Bars */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">Resource Utilization</span>
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
              <span style={{ color: 'var(--text-muted)' }}>CPU Utilization</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{cpuUtil}%</span>
            </div>
            <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${cpuUtil}%`, borderRadius: 4,
                background: cpuUtil > 90 ? 'var(--error)' : cpuUtil > 70 ? 'var(--warning)' : 'var(--accent-secondary)',
                transition: 'width 0.5s ease',
              }} />
            </div>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
              <span style={{ color: 'var(--text-muted)' }}>GPU Utilization</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{gpuUtil}%</span>
            </div>
            <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${gpuUtil}%`, borderRadius: 4,
                background: gpuUtil > 90 ? 'var(--error)' : gpuUtil > 70 ? 'var(--warning)' : 'var(--accent-primary)',
                transition: 'width 0.5s ease',
              }} />
            </div>
          </div>
        </div>
      </div>

      {/* Ray Jobs Table */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">Ray Jobs</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{jobs.length} jobs</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Status</th>
              <th>Entrypoint</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job: any) => {
              const status = (job.status || 'UNKNOWN').toLowerCase();
              const badgeClass = status === 'running' ? 'running' : status === 'succeeded' ? 'completed' : status === 'failed' ? 'failed' : 'idle';
              const startedAgo = job.start_time ? (() => {
                const diff = Date.now() - new Date(job.start_time).getTime();
                if (diff < 3600_000) return `${Math.max(1, Math.round(diff / 60_000))}m ago`;
                if (diff < 86400_000) return `${Math.round(diff / 3600_000)}h ago`;
                return `${Math.round(diff / 86400_000)}d ago`;
              })() : '—';
              return (
                <tr key={job.job_id}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 500 }}>{job.job_id}</td>
                  <td>
                    <span className={`badge ${badgeClass}`}>
                      <span className="badge-dot" />
                      {job.status}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {job.entrypoint || '—'}
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>{startedAgo}</td>
                </tr>
              );
            })}
            {jobs.length === 0 && (
              <tr>
                <td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-dimmed)', padding: 24 }}>
                  No Ray jobs found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Ray Dashboard */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <span className="card-title">Ray Dashboard</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>Embedded via iframe (kiosk mode)</span>
        </div>
        <div className="card-body" style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 200,
          color: 'var(--text-dimmed)',
          fontSize: 14,
        }}>
          <div style={{ textAlign: 'center' }}>
            <Activity style={{ width: 48, height: 48, margin: '0 auto 16px', opacity: 0.3 }} />
            <p>Ray Dashboard iframe will be embedded here when Ray cluster is deployed.</p>
            <p style={{ fontSize: 12, marginTop: 8, fontFamily: 'var(--font-mono)' }}>ray-cluster-head-svc.ray.svc.cluster.local:8265</p>
          </div>
        </div>
      </div>
    </div>
  );
}
