import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-User': 'dashboard',
  },
});

export interface Pod {
  name: string;
  namespace: string;
  status: string;
  node?: string;
  gpu?: number;
  user?: string;
  created_at: string;
}

export interface Job {
  job_id: string;
  workflow: string;
  status: string;
  created_at: string;
  duration?: string;
}

export interface CostReport {
  total_cost: number;
  period_start: string;
  period_end: string;
  jobs: Array<{
    job_id: string;
    workflow: string;
    instance_type: string;
    duration_hours: number;
    cost_usd: number;
    created_at: string;
  }>;
}

export const api = {
  pods: {
    list: async (namespace?: string) => {
      const { data } = await apiClient.get<{ pods: Pod[] }>('/pods', {
        params: { namespace },
      });
      return data.pods;
    },
  },
  jobs: {
    list: async (limit = 50) => {
      const { data } = await apiClient.get<{ jobs: Job[] }>('/jobs', {
        params: { limit },
      });
      return data.jobs;
    },
    get: async (jobId: string) => {
      const { data } = await apiClient.get(`/jobs/${jobId}`);
      return data;
    },
  },
  cost: {
    getReport: async (days = 7) => {
      const { data } = await apiClient.get<CostReport>('/cost/report', {
        params: { days },
      });
      return data;
    },
  },
  dashboard: {
    getOverview: async () => {
      const { data } = await apiClient.get('/dashboard/overview');
      return data;
    },
    getMetrics: async (metric: string, timeframe = '1h') => {
      const { data } = await apiClient.get(`/dashboard/metrics/${metric}`, {
        params: { timeframe },
      });
      return data;
    },
  },
  mlflow: {
    listExperiments: async () => {
      const { data } = await apiClient.get('/mlflow/experiments');
      return data;
    },
    listRuns: async (experimentId: string) => {
      const { data } = await apiClient.get(`/mlflow/experiments/${experimentId}/runs`);
      return data;
    },
    listModels: async () => {
      const { data } = await apiClient.get('/mlflow/models');
      return data;
    },
  },
  ray: {
    getClusterStatus: async () => {
      const { data } = await apiClient.get('/ray/cluster');
      return data;
    },
    getJobs: async () => {
      const { data } = await apiClient.get('/ray/jobs');
      return data;
    },
  },
  kubernetes: {
    listNodes: async () => {
      const { data } = await apiClient.get('/kubernetes/nodes');
      return data;
    },
    listEvents: async (namespace?: string) => {
      const { data } = await apiClient.get('/kubernetes/events', {
        params: { namespace },
      });
      return data;
    },
    getPodLogs: async (podName: string, namespace: string) => {
      const { data } = await apiClient.get(`/kubernetes/pods/${podName}/logs`, {
        params: { namespace },
      });
      return data;
    },
  },
  chat: {
    ask: async (message: string, history: Array<{ role: string; content: string }> = []) => {
      const { data } = await apiClient.post('/chat/ask', { message, history });
      return data;
    },
    listPins: async () => {
      const { data } = await apiClient.get('/chat/pins');
      return data;
    },
    pin: async (payload: { title: string; query: string; widget: Record<string, unknown> }) => {
      const { data } = await apiClient.post('/chat/pins', payload);
      return data;
    },
    unpin: async (pinId: string) => {
      await apiClient.delete(`/chat/pins/${pinId}`);
    },
  },
};
