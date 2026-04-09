import axios from 'axios';
import { mockApi } from './mock-api';
import { mockModelDetail, mockRecipeDetail } from './mock-data';

/** true when running with VITE_MOCK=true (no backend needed) */
export const IS_MOCK = import.meta.env.VITE_MOCK === 'true';

// In Vite, the dev server proxies /api requests to the backend.
const API_BASE_URL = '/api/v1';

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

const _realApi = {
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
    getLogs: async (jobId: string) => {
      const { data } = await apiClient.get(`/jobs/${jobId}/logs`, { responseType: 'text' });
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
  recipes: {
    list: async () => {
      const { data } = await apiClient.get('/recipes');
      return data;
    },
  },
  components: {
    list: async () => {
      const { data } = await apiClient.get('/components');
      return data;
    },
    get: async (name: string) => {
      const { data } = await apiClient.get(`/components/${encodeURIComponent(name)}`);
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
  desks: {
    list: async (user?: string) => {
      const { data } = await apiClient.get('/desks', { params: { user } });
      return data;
    },
    launch: async (spec: { name: string; image?: string; gpu_type?: string; gpu_count?: number; cpu_count?: number }) => {
      const { data } = await apiClient.post('/desks', spec);
      return data;
    },
    stop: async (deskId: string) => {
      const { data } = await apiClient.delete(`/desks/${deskId}`);
      return data;
    },
  },
  serving: {
    listEndpoints: async () => {
      const { data } = await apiClient.get('/serving');
      return data;
    },
    deploy: async (spec: { name: string; model_name: string; model_version: string; gpu_type?: string; replicas?: number; traffic_percent?: number }) => {
      const { data } = await apiClient.post('/serving', spec);
      return data;
    },
    promote: async (endpointName: string, trafficPercent: number = 100) => {
      const { data } = await apiClient.post(`/serving/${endpointName}/promote`, null, { params: { traffic_percent: trafficPercent } });
      return data;
    },
  },
  settings: {
    get: async () => {
      const { data } = await apiClient.get('/settings');
      return data;
    },
    update: async (settings: Record<string, unknown>) => {
      const { data } = await apiClient.put('/settings', settings);
      return data;
    },
    patch: async (updates: Record<string, unknown>) => {
      const { data } = await apiClient.patch('/settings', updates);
      return data;
    },
  },
  tensorboard: {
    getRuns: async () => {
      const { data } = await apiClient.get<{ runs: { execution_id: string; s3_path: string }[] }>('/tensorboard/runs');
      return data;
    },
    getUrl: async (executionId?: string) => {
      const { data } = await apiClient.get<{ url: string }>('/tensorboard/url');
      const url = data.url;
      return executionId ? `${url}/#scalars&regexInput=${encodeURIComponent(executionId)}` : url;
    },
  },
};

/*
 * When VITE_MOCK=true, swap the entire api object for
 * the mock implementation so every page gets fixture data.
 *
 * Usage:
 *   VITE_MOCK=true npm run dev     # no backend needed
 *   npm run dev                    # real backend
 */
export const api: typeof _realApi = IS_MOCK
  ? (mockApi as typeof _realApi)
  : _realApi;

// Named exports for convenience
export const fetchDashboardOverview = api.dashboard.getOverview;
export const fetchCostReport = api.cost.getReport;
export const fetchDesks = api.desks.list;
export const fetchEndpoints = api.serving.listEndpoints;
export const fetchSettings = api.settings.get;

/*
 * When mock mode is active, intercept all direct apiClient calls
 * (e.g. apiClient.get('/mlflow/models/...')) so pages that bypass
 * the `api` object also get fixture data.
 */
if (IS_MOCK) {
  const mockRoutes: Record<string, (url: string, config?: Record<string, unknown>) => unknown> = {
    '/mlflow/models/': (url) => {
      const name = decodeURIComponent(url.split('/mlflow/models/')[1]);
      return mockModelDetail(name);
    },
    '/recipes/': (url) => {
      const name = decodeURIComponent(url.split('/recipes/')[1]);
      return mockRecipeDetail(name);
    },
    '/chat/ask': (_url: string, config?: Record<string, unknown>) => {
      const body = typeof config?.data === 'string' ? JSON.parse(config.data) : config?.data;
      return mockApi.chat.ask(body?.message ?? 'mock', body?.history ?? []);
    },
  };

  apiClient.interceptors.request.use((config) => {
    const url = config.url || '';
    for (const [prefix, handler] of Object.entries(mockRoutes)) {
      if (url.startsWith(prefix)) {
        // Cancel the real request and resolve with mock data
        const source = axios.CancelToken.source();
        config.cancelToken = source.token;
        config.adapter = async () => {
          await new Promise((r) => setTimeout(r, 100));
          const data = typeof handler === 'function'
            ? await handler(url, config as unknown as Record<string, unknown>)
            : handler;
          return { data, status: 200, statusText: 'OK (mock)', headers: {}, config };
        };
        return config;
      }
    }
    return config;
  });
}

