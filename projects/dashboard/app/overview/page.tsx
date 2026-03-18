'use client';

import { useQuery } from '@tanstack/react-query';
import { Activity, DollarSign, Server, TrendingUp } from 'lucide-react';
import { StatCard } from '@/components/StatCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { StatusBadge } from '@/components/StatusBadge';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { api } from '@/lib/api';
import { formatDate, formatCurrency } from '@/lib/utils';

export default function OverviewPage() {
  const { data: pods, isLoading: podsLoading, error: podsError } = useQuery({
    queryKey: ['pods'],
    queryFn: () => api.pods.list(),
  });

  const { data: jobs, isLoading: jobsLoading, error: jobsError } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.jobs.list(10),
  });

  const { data: costReport, isLoading: costLoading, error: costError } = useQuery({
    queryKey: ['cost'],
    queryFn: () => api.cost.getReport(7),
  });

  if (podsLoading || jobsLoading || costLoading) {
    return <LoadingSpinner />;
  }

  if (podsError || jobsError || costError) {
    return <ErrorMessage message="Failed to load overview data" />;
  }

  const activePods = pods?.filter((p) => p.status === 'Running') || [];
  const gpuPods = activePods.filter((p) => p.gpu);
  const recentJobs = jobs || [];
  const totalCost = costReport?.total_cost || 0;

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Active Pods"
          value={activePods.length}
          icon={<Server className="w-6 h-6 text-blue-600" />}
        />
        <StatCard
          title="GPU Pods"
          value={gpuPods.length}
          icon={<Activity className="w-6 h-6 text-green-600" />}
        />
        <StatCard
          title="Recent Jobs"
          value={recentJobs.length}
          icon={<TrendingUp className="w-6 h-6 text-purple-600" />}
        />
        <StatCard
          title="7-Day Cost"
          value={formatCurrency(totalCost)}
          icon={<DollarSign className="w-6 h-6 text-yellow-600" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Active GPU Pods</CardTitle>
          </CardHeader>
          <CardContent>
            {gpuPods.length === 0 ? (
              <p className="text-gray-500 text-sm">No active GPU pods</p>
            ) : (
              <div className="space-y-3">
                {gpuPods.slice(0, 5).map((pod: any) => (
                  <div key={pod.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{pod.name}</p>
                      <p className="text-xs text-gray-500">{pod.namespace}</p>
                    </div>
                    <div className="ml-4 flex items-center space-x-2">
                      {pod.gpu && <span className="text-xs text-gray-600">{pod.gpu}</span>}
                      <StatusBadge status={pod.status} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Executions</CardTitle>
          </CardHeader>
          <CardContent>
            {recentJobs.length === 0 ? (
              <p className="text-gray-500 text-sm">No recent executions</p>
            ) : (
              <div className="space-y-3">
                {recentJobs.slice(0, 5).map((job: any) => (
                  <div key={job.job_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{job.workflow}</p>
                      <p className="text-xs text-gray-500">{formatDate(job.created_at)}</p>
                    </div>
                    <StatusBadge status={job.status} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
