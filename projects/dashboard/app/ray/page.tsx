'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorMessage } from '@/components/ErrorMessage';
import { api } from '@/lib/api';

export default function RayPage() {
  const { data: clusterStatus, isLoading: clusterLoading, error: clusterError } = useQuery({
    queryKey: ['ray-cluster'],
    queryFn: () => api.ray.getClusterStatus(),
  });

  const { data: rayJobs, isLoading: jobsLoading, error: jobsError } = useQuery({
    queryKey: ['ray-jobs'],
    queryFn: () => api.ray.getJobs(),
  });

  if (clusterLoading || jobsLoading) {
    return <LoadingSpinner />;
  }

  if (clusterError || jobsError) {
    return <ErrorMessage message="Failed to load Ray cluster data" />;
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Ray Cluster</h1>
        <p className="text-gray-600 mt-2">Distributed computing cluster status and jobs</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Cluster Status</CardTitle>
          </CardHeader>
          <CardContent>
            {clusterStatus ? (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Active Nodes:</span>
                  <span className="font-medium">{clusterStatus.active_nodes || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total CPUs:</span>
                  <span className="font-medium">{clusterStatus.total_cpus || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total GPUs:</span>
                  <span className="font-medium">{clusterStatus.total_gpus || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Available CPUs:</span>
                  <span className="font-medium">{clusterStatus.available_cpus || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Available GPUs:</span>
                  <span className="font-medium">{clusterStatus.available_gpus || 0}</span>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No cluster information available</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            {rayJobs && rayJobs.length > 0 ? (
              <div className="space-y-3">
                {rayJobs.map((job: any) => (
                  <div key={job.job_id} className="p-3 bg-gray-50 rounded-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-900">{job.job_id}</span>
                      <span className="text-xs text-gray-500">{job.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No active jobs</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
