
import { useQuery } from '@tanstack/react-query';
import { Server, Cpu, Zap, HardDrive, AlertTriangle } from 'lucide-react';
import { api } from '@/lib/api';

interface NodeInfo {
  name: string;
  status: string;
  instance_type: string;
  zone: string;
  capacity: { cpu: string; memory: string; gpu: string };
  allocatable: { cpu: string; memory: string; gpu: string };
}



export default function InfrastructurePage() {
  const { data: nodes } = useQuery<NodeInfo[]>({
    queryKey: ['kubernetes-nodes'],
    queryFn: () => api.kubernetes.listNodes(),
    staleTime: 30_000,
    retry: 2,
  });

  const { data: events } = useQuery({
    queryKey: ['kubernetes-events'],
    queryFn: () => api.kubernetes.listEvents(),
    staleTime: 15_000,
    retry: 2,
  });

  const nodeList = nodes || [];
  const eventList = (events || []).slice(0, 20);

  const readyNodes = nodeList.filter((n) => n.status === 'Ready');
  const totalCpu = nodeList.reduce((sum, n) => sum + parseInt(n.capacity.cpu || '0', 10), 0);
  const totalGpu = nodeList.reduce((sum, n) => sum + parseInt(n.capacity.gpu || '0', 10), 0);
  const totalMemGi = nodeList.reduce((sum, n) => {
    const ki = parseInt((n.capacity.memory || '0').replace(/Ki$/, ''), 10) || 0;
    return sum + Math.round(ki / 1048576);
  }, 0);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Infrastructure</h1>
        <p>Cluster nodes, capacity, and Kubernetes events</p>
      </div>

      {/* Summary Stats */}
      <div className="stat-grid">
        <div className="stat-card">
          <div className="stat-label">Total Nodes</div>
          <div className="stat-value">{readyNodes.length} / {nodeList.length}</div>
          <div className="stat-sub"><Server style={{ width: 12, height: 12 }} /> all healthy</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total CPUs</div>
          <div className="stat-value">{totalCpu}</div>
          <div className="stat-sub"><Cpu style={{ width: 12, height: 12 }} /> cores</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total GPUs</div>
          <div className="stat-value">{totalGpu}</div>
          <div className="stat-sub"><Zap style={{ width: 12, height: 12 }} /> A100/T4</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Memory</div>
          <div className="stat-value">{totalMemGi} Gi</div>
          <div className="stat-sub"><HardDrive style={{ width: 12, height: 12 }} /> cluster-wide</div>
        </div>
      </div>

      {/* Nodes Table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title">Cluster Nodes</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Node</th>
              <th>Status</th>
              <th>Instance</th>
              <th>Zone</th>
              <th>CPU</th>
              <th>Memory</th>
              <th>GPU</th>
            </tr>
          </thead>
          <tbody>
            {nodeList.map((node) => {
              const memKi = parseInt((node.capacity.memory || '0').replace(/Ki$/, ''), 10) || 0;
              const memGi = (memKi / 1048576).toFixed(0);
              const gpuCount = parseInt(node.capacity.gpu || '0', 10);
              return (
                <tr key={node.name}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{node.name}</td>
                  <td><span className="badge completed"><span className="badge-dot" />{node.status}</span></td>
                  <td style={{ fontWeight: 500 }}>{node.instance_type}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{node.zone}</td>
                  <td>{node.capacity.cpu}</td>
                  <td>{memGi} Gi</td>
                  <td>
                    {gpuCount > 0 ? (
                      <span style={{
                        fontSize: 12,
                        fontWeight: 600,
                        color: 'var(--accent-primary)',
                        background: 'rgba(124,58,237,0.12)',
                        padding: '2px 8px',
                        borderRadius: 100,
                      }}>
                        {gpuCount} GPU
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-dimmed)' }}>—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Events */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Events</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>
            {eventList.filter((e: any) => e.type === 'Warning').length} warnings
          </span>
        </div>
        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflowY: 'auto' }}>
          {eventList.map((evt: any, i: number) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
              padding: '10px 14px',
              background: evt.type === 'Warning' ? 'rgba(245,158,11,0.06)' : 'rgba(255,255,255,0.02)',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${evt.type === 'Warning' ? 'rgba(245,158,11,0.15)' : 'var(--border-subtle)'}`,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%', marginTop: 6, flexShrink: 0,
                background: evt.type === 'Warning' ? 'var(--warning)' : 'var(--success)',
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontWeight: 500, fontSize: 13 }}>{evt.reason}</span>
                  <span style={{ color: 'var(--text-dimmed)', fontSize: 11 }}>
                    {evt.involved_object.kind}/{evt.involved_object.name}
                  </span>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {evt.message}
                </p>
                <span style={{ fontSize: 11, color: 'var(--text-dimmed)' }}>
                  {new Date(evt.timestamp).toLocaleString()} · {evt.namespace}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
