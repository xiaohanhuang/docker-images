'use client';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';

export default function InfrastructurePage() {
  const grafanaUrl = process.env.NEXT_PUBLIC_GRAFANA_URL || 'http://grafana.ml-platform.internal';

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Infrastructure</h1>
        <p className="text-gray-600 mt-2">GPU nodes, utilization, and cost trends</p>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>GPU Utilization Dashboard</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <iframe
            src={`${grafanaUrl}/d/gpu-dashboard/gpu-utilization?orgId=1&refresh=30s&kiosk`}
            className="w-full h-[600px] border-0"
            title="GPU Utilization Dashboard"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cluster Metrics</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <iframe
            src={`${grafanaUrl}/d/cluster-metrics/cluster-overview?orgId=1&refresh=30s&kiosk`}
            className="w-full h-[600px] border-0"
            title="Cluster Metrics Dashboard"
          />
        </CardContent>
      </Card>
    </div>
  );
}
