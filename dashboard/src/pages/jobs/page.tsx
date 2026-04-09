
import { useQuery } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import { GitBranch, ExternalLink, Sparkles } from 'lucide-react';
import { api } from '@/lib/api';
import { useChat } from '@/lib/ChatContext';



export default function JobsPage() {
  const navigate = useNavigate();
  const { openChat } = useChat();
  const { data, isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.jobs.list(50),
    staleTime: 30_000,
    retry: 2,
  });

  // Format duration from Flyte format (e.g. "0:59:37.155014") to human readable
  const formatDuration = (dur: string | null | undefined) => {
    if (!dur) return '—';
    // Handle "H:MM:SS.xxx" format
    const match = dur.match(/(\d+):(\d+):(\d+)/);
    if (match) {
      const h = parseInt(match[1]);
      const m = parseInt(match[2]);
      return `${h}h ${m.toString().padStart(2, '0')}m`;
    }
    return dur;
  };

  const jobs = data || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Jobs & Pipelines</h1>
        <p>Monitor and manage your training runs and pipeline executions</p>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">All Jobs</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{jobs.length} total</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>Pipeline</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Duration</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job: any) => {
              const status = (job.status || 'unknown').toLowerCase();
              const statusMap: Record<string, string> = {
                running: 'running', succeeded: 'completed', failed: 'failed',
                completed: 'completed', pending: 'idle', unknown: 'idle',
              };
              const badgeClass = statusMap[status] || 'idle';
              const progress = job.progress ?? (status === 'succeeded' || status === 'completed' ? 100 : status === 'running' ? 50 : 0);
              return (
                <tr key={job.job_id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/jobs/${job.job_id}`)}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <GitBranch style={{ width: 14, height: 14, color: 'var(--text-dimmed)' }} />
                      <span style={{ fontWeight: 500 }}>{job.job_id}</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-muted)' }}>{job.workflow || '—'}</td>
                  <td>
                    <span className={`badge ${badgeClass}`}>
                      <span className="badge-dot" />
                      {status.charAt(0).toUpperCase() + status.slice(1)}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        width: 60, height: 4,
                        background: 'rgba(255,255,255,0.06)',
                        borderRadius: 2,
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${progress}%`,
                          background: badgeClass === 'failed' ? 'var(--error)' : progress === 100 ? 'var(--success)' : 'var(--accent-primary)',
                          borderRadius: 2,
                        }} />
                      </div>
                      <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{progress}%</span>
                    </div>
                  </td>
                  <td>{formatDuration(job.duration)}</td>
                  <td>
                    <button className="btn btn-ghost btn-sm" title="Analyze Run" onClick={(e) => { e.stopPropagation(); openChat(`Analyze job ${job.job_id}`); }}>
                      <Sparkles style={{ width: 14, height: 14 }} />
                    </button>
                    <button className="btn btn-ghost btn-sm" title="Mission Control" onClick={(e) => { e.stopPropagation(); navigate(`/jobs/${job.job_id}`); }}>
                      <ExternalLink style={{ width: 14, height: 14 }} />
                    </button>
                  </td>
                </tr>
              );
            })}
            {jobs.length === 0 && (
              <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dimmed)', padding: 24 }}>No jobs found — submit a workflow with Flyte to see executions here</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
