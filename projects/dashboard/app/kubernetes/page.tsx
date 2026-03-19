'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { StatusBadge } from '@/components/StatusBadge';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { api } from '@/lib/api';
import { formatDate } from '@/lib/utils';

export default function KubernetesPage() {
  const [selectedNamespace, setSelectedNamespace] = useState<string>('default');

  const { data: pods, isLoading: podsLoading } = useQuery({
    queryKey: ['k8s-pods', selectedNamespace],
    queryFn: () => api.pods.list(selectedNamespace),
  });

  const { data: nodes, isLoading: nodesLoading } = useQuery({
    queryKey: ['k8s-nodes'],
    queryFn: () => api.kubernetes.listNodes(),
  });

  const { data: events, isLoading: eventsLoading } = useQuery({
    queryKey: ['k8s-events', selectedNamespace],
    queryFn: () => api.kubernetes.listEvents(selectedNamespace),
  });

  if (podsLoading || nodesLoading || eventsLoading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Kubernetes</h1>
        <p className="text-gray-600 mt-2">Pod, node status, events, and logs</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            {nodes && nodes.length > 0 ? (
              <div className="space-y-3">
                {nodes.map((node: any) => (
                  <div key={node.name} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="text-sm font-medium text-gray-900">{node.name}</p>
                        <p className="text-xs text-gray-500">{node.instance_type}</p>
                      </div>
                      <StatusBadge status={node.status} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No nodes found</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Events</CardTitle>
          </CardHeader>
          <CardContent>
            {events && events.length > 0 ? (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {events.slice(0, 10).map((event: any, idx: number) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium text-gray-900">{event.message}</p>
                    <div className="flex justify-between mt-1">
                      <span className="text-xs text-gray-500">{event.type}</span>
                      <span className="text-xs text-gray-500">{formatDate(event.timestamp)}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No recent events</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>Pods</CardTitle>
            <select
              value={selectedNamespace}
              onChange={(e) => setSelectedNamespace(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="default">default</option>
              <option value="ml-platform-development">ml-platform-development</option>
              <option value="flyte">flyte</option>
              <option value="monitoring">monitoring</option>
            </select>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Node
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {pods?.map((pod) => (
                  <tr key={pod.name} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {pod.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <StatusBadge status={pod.status} />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {pod.node || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(pod.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
