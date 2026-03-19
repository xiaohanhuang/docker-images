'use client';

import { useQuery } from '@tanstack/react-query';
import { Cpu, HardDrive, Server, Zap } from 'lucide-react';
import { StatCard } from '@/components/StatCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { StatusBadge } from '@/components/StatusBadge';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { api } from '@/lib/api';

interface NodeInfo {
  name: string;
  status: string;
  instance_type: string;
  zone: string;
  capacity: { cpu: string; memory: string; gpu: string };
  allocatable: { cpu: string; memory: string; gpu: string };
}

interface EventInfo {
  namespace: string;
  name: string;
  type: string;
  reason: string;
  message: string;
  timestamp: string;
  involved_object: { kind: string; name: string };
}

export default function InfrastructurePage() {
  const { data: nodes, isLoading: nodesLoading } = useQuery<NodeInfo[]>({
    queryKey: ['kubernetes-nodes'],
    queryFn: () => api.kubernetes.listNodes(),
    refetchInterval: 30_000,
  });

  const { data: events, isLoading: eventsLoading } = useQuery<EventInfo[]>({
    queryKey: ['kubernetes-events'],
    queryFn: () => api.kubernetes.listEvents(),
    refetchInterval: 15_000,
  });

  if (nodesLoading) {
    return <LoadingSpinner />;
  }

  const nodeList = nodes || [];
  const readyNodes = nodeList.filter((n) => n.status === 'Ready');
  const totalCpu = nodeList.reduce((sum, n) => sum + parseInt(n.capacity.cpu || '0', 10), 0);
  const totalGpu = nodeList.reduce((sum, n) => sum + parseInt(n.capacity.gpu || '0', 10), 0);
  const totalMemGi = nodeList.reduce((sum, n) => {
    const mem = n.capacity.memory || '0';
    const ki = parseInt(mem.replace(/Ki$/, ''), 10) || 0;
    return sum + Math.round(ki / 1048576);
  }, 0);

  const recentEvents = (events || []).slice(0, 20);
  const warningEvents = recentEvents.filter((e) => e.type === 'Warning');

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Infrastructure</h1>
        <p className="text-gray-600 mt-2">Cluster nodes, capacity, and recent events</p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Nodes"
          value={`${readyNodes.length} / ${nodeList.length}`}
          icon={<Server className="w-6 h-6 text-blue-600" />}
        />
        <StatCard
          title="Total CPU"
          value={`${totalCpu} cores`}
          icon={<Cpu className="w-6 h-6 text-green-600" />}
        />
        <StatCard
          title="Total GPU"
          value={totalGpu}
          icon={<Zap className="w-6 h-6 text-purple-600" />}
        />
        <StatCard
          title="Total Memory"
          value={`${totalMemGi} Gi`}
          icon={<HardDrive className="w-6 h-6 text-yellow-600" />}
        />
      </div>

      {/* Nodes table */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Cluster Nodes</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">Name</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">Status</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">Instance</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">Zone</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">CPU</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">Memory</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">GPU</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {nodeList.map((node) => {
                  const memKi = parseInt((node.capacity.memory || '0').replace(/Ki$/, ''), 10) || 0;
                  const memGi = (memKi / 1048576).toFixed(1);
                  return (
                    <tr key={node.name} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-gray-900 font-mono text-xs">{node.name}</td>
                      <td className="px-4 py-3"><StatusBadge status={node.status} /></td>
                      <td className="px-4 py-3 text-gray-700">{node.instance_type}</td>
                      <td className="px-4 py-3 text-gray-500">{node.zone}</td>
                      <td className="px-4 py-3 text-gray-700">{node.capacity.cpu}</td>
                      <td className="px-4 py-3 text-gray-700">{memGi} Gi</td>
                      <td className="px-4 py-3 text-gray-700">
                        {parseInt(node.capacity.gpu || '0', 10) > 0 ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                            {node.capacity.gpu} GPU
                          </span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Recent Events */}
      <Card>
        <CardHeader>
          <CardTitle>
            Recent Events
            {warningEvents.length > 0 && (
              <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                {warningEvents.length} warnings
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {eventsLoading ? (
            <p className="text-sm text-gray-500">Loading events...</p>
          ) : recentEvents.length === 0 ? (
            <p className="text-sm text-gray-500">No recent events</p>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {recentEvents.map((evt, i) => (
                <div
                  key={`${evt.name}-${i}`}
                  className={`flex items-start gap-3 p-3 rounded-lg text-sm ${
                    evt.type === 'Warning' ? 'bg-yellow-50' : 'bg-gray-50'
                  }`}
                >
                  <span
                    className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${
                      evt.type === 'Warning' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900">{evt.reason}</span>
                      <span className="text-gray-400">-</span>
                      <span className="text-gray-500 text-xs">{evt.involved_object.kind}/{evt.involved_object.name}</span>
                    </div>
                    <p className="text-gray-600 text-xs mt-0.5 truncate">{evt.message}</p>
                    <p className="text-gray-400 text-xs mt-0.5">
                      {new Date(evt.timestamp).toLocaleString()} &middot; {evt.namespace}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
