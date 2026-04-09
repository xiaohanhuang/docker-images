/**
 * Mock API — drop-in replacement for the real `api` object.
 *
 * Every method signature matches the real api in lib/api.ts but returns
 * fixture data from lib/mock-data.ts after a short simulated delay.
 *
 * Activate with:  VITE_MOCK=true
 */
import type { Pod, Job, CostReport } from './api';
import * as mock from './mock-data';

/** Simulate realistic network latency (80-250 ms). */
const delay = (ms = 80 + Math.random() * 170) =>
  new Promise<void>((r) => setTimeout(r, ms));

export const mockApi = {
  pods: {
    list: async (_namespace?: string): Promise<Pod[]> => {
      await delay();
      return mock.mockPods;
    },
  },

  jobs: {
    list: async (_limit = 50): Promise<Job[]> => {
      await delay();
      return mock.mockJobs;
    },
    get: async (jobId: string) => {
      await delay();
      return mock.mockJobs.find((j) => j.job_id === jobId) || mock.mockJobs[0];
    },
    getLogs: async (_jobId: string) => {
      await delay();
      return mock.mockJobLogs;
    },
  },

  cost: {
    getReport: async (_days = 7): Promise<CostReport> => {
      await delay();
      return mock.mockCostReport;
    },
  },

  recipes: {
    list: async () => {
      await delay();
      return mock.mockRecipes;
    },
  },

  dashboard: {
    getOverview: async () => {
      await delay();
      return mock.mockDashboardOverview;
    },
    getMetrics: async (_metric: string, _timeframe = '1h') => {
      await delay();
      return {
        metric: _metric,
        values: Array.from({ length: 30 }, (_, i) => ({
          timestamp: Date.now() - (29 - i) * 120_000,
          value: 40 + Math.random() * 50,
        })),
      };
    },
  },

  mlflow: {
    listExperiments: async () => {
      await delay();
      return mock.mockExperiments;
    },
    listRuns: async (_experimentId: string) => {
      await delay();
      return mock.mockRuns;
    },
    listModels: async () => {
      await delay();
      return mock.mockModels;
    },
  },

  ray: {
    getClusterStatus: async () => {
      await delay();
      return mock.mockRayCluster;
    },
    getJobs: async () => {
      await delay();
      return mock.mockRayJobs;
    },
  },

  kubernetes: {
    listNodes: async () => {
      await delay();
      return mock.mockNodes;
    },
    listEvents: async (_namespace?: string) => {
      await delay();
      return mock.mockEvents;
    },
    getPodLogs: async (_podName: string, _namespace: string) => {
      await delay();
      return mock.mockJobLogs;
    },
  },

  chat: {
    ask: async (message: string, _history: Array<{ role: string; content: string }> = []) => {
      await delay(300);
      return {
        text: `[Mock] Answering: "${message}". In mock mode, chat responses are simulated.`,
        widget: {
          type: 'stat' as const,
          title: 'Mock Response',
          value: '42',
          unit: 'items',
          trend: 12.5,
        },
      };
    },
    listPins: async () => {
      await delay();
      return mock.mockPins;
    },
    pin: async (payload: { title: string; query: string; widget: Record<string, unknown> }) => {
      await delay();
      return { id: `pin-${Date.now()}`, ...payload, created_at: new Date().toISOString() };
    },
    unpin: async (_pinId: string) => {
      await delay();
    },
  },

  desks: {
    list: async (_user?: string) => {
      await delay();
      return mock.mockDesks;
    },
    launch: async (spec: { name: string; image?: string; gpu_type?: string; gpu_count?: number; cpu_count?: number }) => {
      await delay(500);
      return { status: 'launched', desk_id: `desk-${Date.now()}`, message: `Desk ${spec.name} created (mock)` };
    },
    stop: async (deskId: string) => {
      await delay(300);
      return { status: 'stopped', desk_id: deskId, message: `Desk ${deskId} stopped (mock)` };
    },
  },

  serving: {
    listEndpoints: async () => {
      await delay();
      return mock.mockServing;
    },
    deploy: async (spec: { name: string; model_name: string; model_version: string; gpu_type?: string; replicas?: number; traffic_percent?: number }) => {
      await delay(500);
      return { status: 'deployed', name: spec.name, message: `Endpoint ${spec.name} deployed (mock)` };
    },
    promote: async (endpointName: string, _trafficPercent: number = 100) => {
      await delay(300);
      return { status: 'promoted', name: endpointName, message: `Traffic updated (mock)` };
    },
  },

  components: {
    list: async () => {
      await delay();
      return {
        components: [
          { name: 'hf_dataset_loader', version: '1.9.5', desc: 'Load datasets from HuggingFace Hub', type: 'task', category: 'data', tags: ['data', 'huggingface', 'datasets'] },
          { name: 'tokenizer', version: '1.4.1', desc: 'Tokenize text data with prompt templates', type: 'task', category: 'data', tags: ['data', 'tokenization', 'llm'] },
          { name: 'lora_finetune', version: '1.9.22', desc: 'LoRA/QLoRA fine-tuning for HuggingFace models', type: 'task', category: 'training', tags: ['training', 'lora', 'qlora', 'fine-tuning'] },
          { name: 'llm_judge', version: '1.11.5', desc: 'LLM-as-Judge for evaluating model outputs', type: 'task', category: 'evaluation', tags: ['evaluation', 'llm-judge', 'alignment'] },
          { name: 'registry_publisher', version: '1.10.6', desc: 'Register fine-tuned models in MLflow', type: 'task', category: 'model', tags: ['model', 'mlflow', 'registry'] },
          { name: 'distributed_rlhf_trainer', version: '1.10.16', desc: 'Distributed multi-role RLHF training using OpenRLHF', type: 'task', category: 'training', tags: ['training', 'rlhf', 'distributed', 'openrlhf'] },
          { name: 'vllm_deployer', version: '1.3.1', desc: 'Deploy a HuggingFace model as a vLLM inference endpoint', type: 'task', category: 'serving', tags: ['serving', 'vllm', 'deployment'] },
          { name: 'text_chunker', version: '1.3.1', desc: 'Split documents into overlapping chunks for RAG indexing', type: 'task', category: 'data', tags: ['data', 'rag', 'chunking'] },
        ],
      };
    },
    get: async (name: string) => {
      await delay();
      const details: Record<string, unknown> = {
        hf_dataset_loader: { name: 'hf_dataset_loader', version: '1.9.5', desc: 'Load datasets from HuggingFace Hub', category: 'data', tags: ['data', 'huggingface', 'datasets'], image: '805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/ml-gpu:1.1.1', image_tag: '1.1.1', task_type: 'python-task', inputs: [{ name: 'dataset_name', type: 'str' }, { name: 'split', type: 'str' }, { name: 'max_samples', type: 'Optional[int]' }], outputs: [{ name: 'dataset_path', type: 'FlyteDirectory' }, { name: 'num_rows', type: 'int' }] },
        tokenizer: { name: 'tokenizer', version: '1.4.1', desc: 'Tokenize text data with prompt templates', category: 'data', tags: ['data', 'tokenization', 'llm'], image: '805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/data-cpu:1.0.0', image_tag: '1.0.0', task_type: 'python-task', inputs: [{ name: 'input_data', type: 'FlyteFile' }, { name: 'model_name', type: 'str' }, { name: 'max_length', type: 'int' }], outputs: [{ name: 'tokenized_data', type: 'FlyteFile' }, { name: 'vocab_size', type: 'int' }] },
        lora_finetune: { name: 'lora_finetune', version: '1.9.22', desc: 'LoRA/QLoRA fine-tuning for HuggingFace models', category: 'training', tags: ['training', 'lora', 'qlora', 'fine-tuning'], image: '805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/ml-gpu:1.1.1', image_tag: '1.1.1', task_type: 'python-task', inputs: [{ name: 'base_model', type: 'str' }, { name: 'train_data', type: 'FlyteFile' }, { name: 'lora_rank', type: 'int' }, { name: 'learning_rate', type: 'float' }, { name: 'num_epochs', type: 'int' }], outputs: [{ name: 'checkpoint_dir', type: 'FlyteDirectory' }, { name: 'metrics', type: 'Dict[str, float]' }] },
      };
      return details[name] || { name, version: '0.0.0', desc: 'Unknown component', category: '', tags: [], image: '', image_tag: '', task_type: '', inputs: [], outputs: [] };
    },
  },

  settings: {
    get: async () => {
      await delay();
      return { ...mock.mockSettings };
    },
    update: async (settings: Record<string, unknown>) => {
      await delay();
      return settings;
    },
    patch: async (updates: Record<string, unknown>) => {
      await delay();
      return { ...mock.mockSettings, ...updates };
    },
  },
  tensorboard: {
    getRuns: async () => {
      await delay();
      return {
        runs: [
          { execution_id: 'sft-llama3-20260401-001', s3_path: 's3://ml-platform/tensorboard/sft-llama3-20260401-001/' },
          { execution_id: 'dpo-mistral-20260329-002', s3_path: 's3://ml-platform/tensorboard/dpo-mistral-20260329-002/' },
          { execution_id: 'ppo-gpt2-20260325-003', s3_path: 's3://ml-platform/tensorboard/ppo-gpt2-20260325-003/' },
        ],
      };
    },
    getUrl: async (executionId?: string) => {
      await delay();
      const base = 'http://localhost:6006';
      return executionId
        ? `${base}/#scalars&regexInput=${encodeURIComponent(executionId)}`
        : base;
    },
  },
};
