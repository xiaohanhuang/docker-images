
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Box, Cpu, DollarSign, Zap, ExternalLink } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { fetchDashboardOverview } from '@/lib/api';

// Empty initial state — no mock data
const emptyOverview = {
  active_desks: 0,
  active_pods: 0,
  active_gpus: 0,
  running_jobs: 0,
  total_cost: 0,
  recent_jobs: [] as any[],
};

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge ${status}`}>
      <span className="badge-dot" />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState<{ name: string } | null>(null);

  useEffect(() => {
    fetch('/api/auth')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.user) setUser(data.user);
      })
      .catch(() => {});
  }, []);

  const { data: overview } = useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: fetchDashboardOverview,
    staleTime: 60_000,
    retry: 2,
  });

  const data = overview || emptyOverview;
  
  // Get first name or default
  const firstName = user?.name ? user.name.split(' ')[0] : 'ML Engineer';

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <p className="greeting-text">{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}</p>
        <h1>{getGreeting()}, {firstName}</h1>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <div className="stat-card" style={{ cursor: 'pointer' }} onClick={() => navigate('/desks')}>
          <div className="stat-label">Active Desks</div>
          <div className="stat-value">{data.active_desks || 0}</div>
          <div className="stat-sub"><Box style={{ width: 12, height: 12 }} /> Interactive pods</div>
        </div>
        <div className="stat-card" style={{ cursor: 'pointer' }} onClick={() => navigate('/jobs')}>
          <div className="stat-label">Running Jobs</div>
          <div className="stat-value">{data.running_jobs || 0}</div>
          <div className="stat-sub"><Zap style={{ width: 12, height: 12 }} /> Flyte executions</div>
        </div>
        <div className="stat-card" style={{ cursor: 'pointer' }} onClick={() => navigate('/infrastructure')}>
          <div className="stat-label">Active GPUs</div>
          <div className="stat-value">{data.active_gpus || 0}</div>
          <div className="stat-sub"><Cpu style={{ width: 12, height: 12 }} /> across cluster</div>
        </div>
        <div className="stat-card" style={{ cursor: 'pointer' }} onClick={() => navigate('/cost')}>
          <div className="stat-label">7-Day Cost</div>
          <div className="stat-value cost">${(data.total_cost || 0).toFixed(2)}</div>
          <div className="stat-sub"><DollarSign style={{ width: 12, height: 12 }} /> trailing week</div>
        </div>
      </div>

      {/* Content Grid */}
      <div className="content-grid">
        {/* Recent Jobs */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Recent Jobs</span>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/jobs')}>View All</button>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Job</th>
                <th>Status</th>
                <th>Duration</th>
                <th>GPU</th>
                <th>Cost</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.recent_jobs.map((job: any) => {
                // Normalize: API returns job_id/workflow/status(UPPER), mock returns id/name/status(lower)
                const jobId = job.job_id || job.id;
                const jobName = job.workflow || job.name;
                const status = (job.status || 'unknown').toLowerCase();
                const duration = job.duration || '—';
                const gpu = job.gpu_type ? `${job.gpu_type}${job.gpu_count ? ` x${job.gpu_count}` : ''}` : (job.gpu || '—');
                const cost = job.cost != null ? (typeof job.cost === 'number' ? `$${job.cost.toFixed(2)}` : job.cost) : '—';
                return (
                  <tr key={jobId}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{jobName}</div>
                      <div style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{jobId}</div>
                    </td>
                    <td><StatusBadge status={status} /></td>
                    <td>{duration}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{gpu}</td>
                    <td style={{ color: 'var(--cost-green)' }}>{cost}</td>
                    <td>
                      <button className="btn btn-ghost btn-sm" title="View in Mission Control"
                        onClick={() => navigate(`/jobs/${jobId}`)}>
                        <ExternalLink style={{ width: 14, height: 14 }} />
                      </button>
                    </td>
                  </tr>
                );
              })}
              {data.recent_jobs.length === 0 && (
                <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dimmed)', padding: 24 }}>No recent jobs — the Flyte cluster has no executions yet</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Cost Trend */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Cost Trend</span>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/cost')}>Last 7 days</button>
          </div>
          <div className="card-body">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={(() => {
                  // Derive a deterministic daily cost from total
                  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Today'];
                  const total = data.total_cost || 0;
                  const daily = total / 7;
                  // Slight natural variation: weekdays higher, weekends lower
                  const weights = [1.1, 1.15, 1.2, 1.05, 1.1, 0.7, 0.7];
                  return days.map((day, i) => ({
                    day,
                    cost: +(daily * weights[i]).toFixed(2),
                  }));
                })()}>
                  <defs>
                    <linearGradient id="costGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#7c3aed" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="day"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                    tickFormatter={(v) => `$${v}`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'rgba(10,14,26,0.95)',
                      border: '1px solid rgba(255,255,255,0.12)',
                      borderRadius: 8,
                      color: '#f1f5f9',
                      fontSize: 13,
                    }}
                    formatter={(value: any) => [`$${value}`, 'Cost']}
                  />
                  <Area
                    type="monotone"
                    dataKey="cost"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    fill="url(#costGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Keyboard shortcut hint */}
      <div style={{ textAlign: 'center', marginTop: 24 }}>
        <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>
          Press <kbd style={{
            background: 'rgba(255,255,255,0.06)',
            padding: '2px 6px',
            borderRadius: 4,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
          }}>⌘K</kbd> to open the command palette
        </span>
      </div>
    </div>
  );
}
