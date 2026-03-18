import axios from 'axios';

const client = axios.create({
  baseURL: '/api',
  timeout: 15000,
});

export const api = {
  pods: {
    list: async (namespace?: string) => {
      const params = namespace ? { namespace } : {};
      const { data } = await client.get('/pods', { params });
      return data.pods ?? data;
    },
  },
  jobs: {
    list: async (limit: number = 10) => {
      const { data } = await client.get('/jobs', { params: { limit } });
      return data.jobs ?? data;
    },
  },
  cost: {
    getReport: async (days: number = 7) => {
      const { data } = await client.get('/cost/report', { params: { days } });
      return data;
    },
  },
  mlflow: {
    listExperiments: async () => {
      const { data } = await client.get('/mlflow/experiments');
      return data;
    },
    listModels: async () => {
      const { data } = await client.get('/mlflow/models');
      return data;
    },
  },
  ray: {
    getClusterStatus: async () => {
      const { data } = await client.get('/ray/cluster');
      return data;
    },
    getJobs: async () => {
      const { data } = await client.get('/ray/jobs');
      return data;
    },
  },
  kubernetes: {
    listNodes: async () => {
      const { data } = await client.get('/kubernetes/nodes');
      return data;
    },
    listEvents: async (namespace?: string) => {
      const params = namespace ? { namespace } : {};
      const { data } = await client.get('/kubernetes/events', { params });
      return data;
    },
  },
  dashboard: {
    getOverview: async () => {
      const { data } = await client.get('/dashboard/overview');
      return data;
    },
  },
};
