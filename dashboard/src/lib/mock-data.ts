/**
 * Mock data fixtures for local development without a backend.
 *
 * Enable with:  VITE_MOCK=true npm run dev
 */

// ── Helpers ──────────────────────────────────────────────────────

const ago = (hours: number) =>
  new Date(Date.now() - hours * 3600_000).toISOString();

// ── Pods ─────────────────────────────────────────────────────────

export const mockPods = [
  {
    name: 'sft-llama3-8b-run1',
    namespace: 'ml-platform',
    status: 'Running',
    node: 'ip-10-0-3-42.us-west-2.compute.internal',
    gpu: 4,
    image: 'ml-platform/gpu-train:1.1.1',
    user: 'xiaohan',
    created_at: ago(18),
  },
  {
    name: 'eval-text2sql-v3',
    namespace: 'ml-platform',
    status: 'Running',
    node: 'ip-10-0-3-77.us-west-2.compute.internal',
    gpu: 1,
    image: 'ml-platform/gpu-eval:1.1.0',
    user: 'alice',
    created_at: ago(4),
  },
  {
    name: 'data-preprocess-batch',
    namespace: 'ml-platform',
    status: 'Succeeded',
    node: 'ip-10-0-1-12.us-west-2.compute.internal',
    gpu: 0,
    image: 'ml-platform/cpu-spark:1.2.0',
    user: 'bob',
    created_at: ago(72),
  },
  {
    name: 'reward-model-train',
    namespace: 'ml-platform',
    status: 'Running',
    node: 'ip-10-0-3-42.us-west-2.compute.internal',
    gpu: 8,
    image: 'ml-platform/gpu-train:1.1.1',
    user: 'xiaohan',
    created_at: ago(6),
  },
];

// ── Jobs ─────────────────────────────────────────────────────────

export const mockJobs = [
  {
    job_id: 'f-abc12345',
    workflow: 'text2sql_pipeline',
    status: 'SUCCEEDED',
    created_at: ago(24),
    started_at: ago(23.5),
    duration: '2h 15m',
    instance_type: 'g5.2xlarge',
    gpu_type: 'A10G',
    gpu_count: 1,
    cost: 4.5,
    error: null,
  },
  {
    job_id: 'f-def67890',
    workflow: 'llm_sft_finetune',
    status: 'RUNNING',
    created_at: ago(3),
    started_at: ago(2.8),
    duration: '2h 48m',
    instance_type: 'g5.12xlarge',
    gpu_type: 'A10G',
    gpu_count: 4,
    cost: 18.2,
    error: null,
  },
  {
    job_id: 'f-ghi11111',
    workflow: 'reward_model_training',
    status: 'RUNNING',
    created_at: ago(6),
    started_at: ago(5.9),
    duration: '5h 54m',
    instance_type: 'p4d.24xlarge',
    gpu_type: 'A100',
    gpu_count: 8,
    cost: 192.0,
    error: null,
  },
  {
    job_id: 'f-jkl22222',
    workflow: 'spark_etl_ingest',
    status: 'SUCCEEDED',
    created_at: ago(48),
    started_at: ago(47.5),
    duration: '45m',
    instance_type: 'r6i.4xlarge',
    gpu_type: null,
    gpu_count: 0,
    cost: 2.1,
    error: null,
  },
  {
    job_id: 'f-mno33333',
    workflow: 'text2sql_pipeline',
    status: 'FAILED',
    created_at: ago(72),
    started_at: ago(71.8),
    duration: '12m',
    instance_type: 'g5.2xlarge',
    gpu_type: 'A10G',
    gpu_count: 1,
    cost: 0.4,
    error: 'OOM: CUDA out of memory',
  },
];

export const mockJobLogs = `[2026-04-01T08:00:00Z] Starting text2sql_pipeline...
[2026-04-01T08:00:01Z] Loading dataset from s3://ml-platform-data/text2sql/v3
[2026-04-01T08:00:15Z] Dataset loaded: 125,000 examples
[2026-04-01T08:00:16Z] Initializing model: meta-llama/Llama-3-8B
[2026-04-01T08:01:30Z] Model loaded on 1x A10G GPU
[2026-04-01T08:01:31Z] Starting training: 3 epochs, batch_size=16, lr=2e-5
[2026-04-01T08:01:31Z] Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━ 100% loss=1.342
[2026-04-01T09:12:00Z] Epoch 2/3 ━━━━━━━━━━━━━━━━━━━━ 100% loss=0.876
[2026-04-01T10:10:00Z] Epoch 3/3 ━━━━━━━━━━━━━━━━━━━━ 100% loss=0.654
[2026-04-01T10:15:00Z] Saving checkpoint to s3://ml-platform-data/checkpoints/text2sql-v3
[2026-04-01T10:15:30Z] Pipeline completed successfully.`;

// ── Cost ─────────────────────────────────────────────────────────

export const mockCostReport = {
  total_cost: 217.2,
  period_start: ago(168),
  period_end: new Date().toISOString(),
  jobs: mockJobs.map((j) => ({
    job_id: j.job_id,
    workflow: j.workflow,
    instance_type: j.instance_type,
    duration_hours: parseFloat(j.duration) || 2.0,
    cost_usd: j.cost,
    created_at: j.created_at,
  })),
};

// ── Dashboard overview ───────────────────────────────────────────

export const mockDashboardOverview = {
  active_desks: 3,
  active_pods: mockPods.filter((p) => p.status === 'Running').length,
  active_gpus: mockPods
    .filter((p) => p.status === 'Running')
    .reduce((s, p) => s + (p.gpu || 0), 0),
  gpu_pods: mockPods.filter((p) => p.status === 'Running' && (p.gpu || 0) > 0).length,
  running_jobs: mockJobs.filter((j) => j.status === 'RUNNING').length,
  recent_jobs: mockJobs.slice(0, 5).map((j) => ({
    job_id: j.job_id,
    workflow: j.workflow,
    status: j.status,
    duration: j.duration,
    gpu_type: j.gpu_type,
    gpu_count: j.gpu_count,
    cost: j.cost,
  })),
  total_cost: mockCostReport.total_cost,
  pods: mockPods,
};

// ── Desks ────────────────────────────────────────────────────────

export const mockDesks = {
  desks: [
    {
      id: 'desk-xh-01',
      name: 'xiaohan-dev',
      status: 'Running',
      gpu: 'A10G',
      cpu_count: 8,
      memory: '32Gi',
      uptime: '14h 22m',
      burn_rate: 1.65,
      image: 'ml-platform/gpu-dev:1.1.0',
      user: 'xiaohan',
      created_at: ago(14),
    },
    {
      id: 'desk-al-02',
      name: 'alice-notebook',
      status: 'Running',
      gpu: null,
      cpu_count: 4,
      memory: '16Gi',
      uptime: '3h 10m',
      burn_rate: 0.42,
      image: 'ml-platform/cpu-jupyter:1.1.0',
      user: 'alice',
      created_at: ago(3),
    },
    {
      id: 'desk-bb-03',
      name: 'bob-debug',
      status: 'Stopped',
      gpu: 'A10G',
      cpu_count: 8,
      memory: '32Gi',
      uptime: '0m',
      burn_rate: 0,
      image: 'ml-platform/gpu-dev:1.1.0',
      user: 'bob',
      created_at: ago(48),
    },
  ],
  count: 3,
};

// ── MLflow ───────────────────────────────────────────────────────

export const mockExperiments = {
  experiments: [
    {
      experiment_id: '1',
      name: 'text2sql-finetune',
      lifecycle_stage: 'active',
      runs: 12,
      best_metric: 0.876,
      last_update_time: ago(4),
      creation_time: ago(720),
    },
    {
      experiment_id: '2',
      name: 'reward-model-v2',
      lifecycle_stage: 'active',
      runs: 8,
      best_metric: 0.921,
      last_update_time: ago(6),
      creation_time: ago(360),
    },
    {
      experiment_id: '3',
      name: 'llm-sft-llama3',
      lifecycle_stage: 'active',
      runs: 24,
      best_metric: 0.834,
      last_update_time: ago(1),
      creation_time: ago(168),
    },
  ],
};

export const mockRuns = {
  runs: [
    {
      info: {
        run_id: 'run-001',
        run_name: 'lr-2e5-bs16',
        status: 'FINISHED',
        start_time: Date.now() - 86400_000,
        end_time: Date.now() - 78000_000,
      },
      data: {
        params: [
          { key: 'learning_rate', value: '2e-5' },
          { key: 'batch_size', value: '16' },
          { key: 'epochs', value: '3' },
        ],
        metrics: [
          { key: 'loss', value: 0.654 },
          { key: 'accuracy', value: 0.876 },
        ],
      },
    },
    {
      info: {
        run_id: 'run-002',
        run_name: 'lr-5e5-bs32',
        status: 'FINISHED',
        start_time: Date.now() - 172800_000,
        end_time: Date.now() - 165600_000,
      },
      data: {
        params: [
          { key: 'learning_rate', value: '5e-5' },
          { key: 'batch_size', value: '32' },
          { key: 'epochs', value: '5' },
        ],
        metrics: [
          { key: 'loss', value: 0.721 },
          { key: 'accuracy', value: 0.843 },
        ],
      },
    },
  ],
};

export const mockModels = {
  registered_models: [
    {
      name: 'text2sql-llama3-8b',
      version: '3',
      stage: 'Production',
      metrics: 'accuracy: 0.876',
      updated: ago(24),
      latest_versions: [
        {
          version: '3',
          current_stage: 'Production',
          creation_timestamp: Date.now() - 86400_000,
          run_id: 'run-001',
          source: 's3://ml-platform-data/models/text2sql-v3',
        },
      ],
    },
    {
      name: 'reward-model-v2',
      version: '2',
      stage: 'Staging',
      metrics: 'accuracy: 0.921',
      updated: ago(48),
      latest_versions: [
        {
          version: '2',
          current_stage: 'Staging',
          creation_timestamp: Date.now() - 172800_000,
          run_id: 'run-003',
          source: 's3://ml-platform-data/models/reward-v2',
        },
      ],
    },
  ],
};

export const mockModelDetail = (name: string) => ({
  registered_model: {
    name,
    description: `Fine-tuned ${name} model for production inference.`,
    tags: [
      { key: 'framework', value: 'pytorch' },
      { key: 'task', value: 'text-to-sql' },
    ],
    latest_versions: mockModels.registered_models.find((m) => m.name === name)
      ?.latest_versions || [
      {
        version: '1',
        current_stage: 'None',
        creation_timestamp: Date.now(),
        run_id: 'run-000',
        source: 's3://ml-platform-data/models/unknown',
      },
    ],
  },
});

// ── Infrastructure / Kubernetes ──────────────────────────────────

export const mockNodes = [
  {
    name: 'ip-10-0-3-42.us-west-2.compute.internal',
    status: 'Ready',
    instance_type: 'g5.12xlarge',
    zone: 'us-west-2a',
    capacity: { cpu: '48', memory: '192Gi', gpu: '4' },
    allocatable: { cpu: '47', memory: '188Gi', gpu: '4' },
  },
  {
    name: 'ip-10-0-3-77.us-west-2.compute.internal',
    status: 'Ready',
    instance_type: 'g5.2xlarge',
    zone: 'us-west-2b',
    capacity: { cpu: '8', memory: '32Gi', gpu: '1' },
    allocatable: { cpu: '7', memory: '30Gi', gpu: '1' },
  },
  {
    name: 'ip-10-0-1-12.us-west-2.compute.internal',
    status: 'Ready',
    instance_type: 'r6i.4xlarge',
    zone: 'us-west-2a',
    capacity: { cpu: '16', memory: '128Gi', gpu: '0' },
    allocatable: { cpu: '15', memory: '124Gi', gpu: '0' },
  },
  {
    name: 'ip-10-0-1-88.us-west-2.compute.internal',
    status: 'Ready',
    instance_type: 'p4d.24xlarge',
    zone: 'us-west-2b',
    capacity: { cpu: '96', memory: '1152Gi', gpu: '8' },
    allocatable: { cpu: '94', memory: '1146Gi', gpu: '8' },
  },
];

export const mockEvents = [
  {
    namespace: 'ml-platform',
    name: 'sft-llama3-8b-run1',
    type: 'Normal',
    reason: 'Scheduled',
    message: 'Successfully assigned ml-platform/sft-llama3-8b-run1 to ip-10-0-3-42',
    timestamp: ago(18),
    involved_object: { kind: 'Pod', name: 'sft-llama3-8b-run1' },
  },
  {
    namespace: 'ml-platform',
    name: 'reward-model-train',
    type: 'Normal',
    reason: 'Pulled',
    message: 'Container image "ml-platform/gpu-train:1.1.1" already present',
    timestamp: ago(6),
    involved_object: { kind: 'Pod', name: 'reward-model-train' },
  },
  {
    namespace: 'kube-system',
    name: 'karpenter-controller',
    type: 'Normal',
    reason: 'ProvisionedNode',
    message: 'Launched instance i-0abc123 (g5.12xlarge) for pods requiring nvidia.com/gpu',
    timestamp: ago(19),
    involved_object: { kind: 'Deployment', name: 'karpenter' },
  },
  {
    namespace: 'ml-platform',
    name: 'data-preprocess-batch',
    type: 'Normal',
    reason: 'Completed',
    message: 'Job completed successfully',
    timestamp: ago(71),
    involved_object: { kind: 'Pod', name: 'data-preprocess-batch' },
  },
];

// ── Ray ──────────────────────────────────────────────────────────

export const mockRayCluster = {
  active_nodes: 3,
  total_cpus: 64,
  total_gpus: 8,
  available_cpus: 22,
  available_gpus: 2,
};

export const mockRayJobs = [
  {
    job_id: 'ray-001',
    status: 'RUNNING',
    entrypoint: 'python train.py --model llama3-8b',
    start_time: ago(2),
  },
  {
    job_id: 'ray-002',
    status: 'SUCCEEDED',
    entrypoint: 'python eval.py --checkpoint latest',
    start_time: ago(26),
  },
];

// ── Serving ──────────────────────────────────────────────────────

export const mockServing = {
  endpoints: [
    {
      name: 'text2sql-prod',
      model: 'text2sql-llama3-8b:v3',
      status: 'Active',
      traffic: 100,
      latency_p99: 142,
      rps: 38.5,
      replicas: 2,
    },
    {
      name: 'reward-staging',
      model: 'reward-model-v2:v2',
      status: 'Shadow',
      traffic: 50,
      latency_p99: 89,
      rps: 12.1,
      replicas: 1,
    },
  ],
  count: 2,
};

// ── Recipes ──────────────────────────────────────────────────────

export const mockRecipes = {
  recipes: [
    {
      name: 'text2sql',
      version: '1.2.0',
      description: 'Text-to-SQL pipeline: ingest → preprocess → train → evaluate → serve',
      author: 'xiaohan',
      tags: ['text2sql', 'nlp', 'fine-tuning'],
      verified: true,
    },
    {
      name: 'llm-sft',
      version: '0.9.0',
      description: 'Supervised fine-tuning for large language models',
      author: 'team-ml',
      tags: ['llm', 'sft', 'fine-tuning'],
      verified: true,
    },
    {
      name: 'reward-model-training',
      version: '0.5.0',
      description: 'Train a reward model for RLHF',
      author: 'xiaohan',
      tags: ['rlhf', 'reward-model'],
      verified: false,
    },
  ],
};

export const mockRecipeDetail = (name: string) => {
  const base = mockRecipes.recipes.find((r) => r.name === name) || {
    name,
    version: '0.1.0',
    description: 'Unknown recipe',
    author: 'unknown',
    tags: [],
    verified: false,
  };
  return {
    ...base,
    steps: [
      { name: 'ingest', description: 'Load and validate dataset from S3' },
      { name: 'preprocess', description: 'Tokenize and format training data' },
      { name: 'train', description: 'Fine-tune model with distributed training' },
      { name: 'evaluate', description: 'Run evaluation metrics on validation set' },
    ],
    profiles: [
      {
        name: 'dev',
        gpu: '1x A10G',
        cost: '$2/hr',
        desc: 'Development — quick iteration',
        ram: '32Gi',
        vram: '24Gi',
      },
      {
        name: 'prod',
        gpu: '4x A10G',
        cost: '$8/hr',
        desc: 'Production — full training run',
        ram: '192Gi',
        vram: '96Gi',
      },
    ],
    params: [
      { key: 'num_epochs', label: 'Epochs', type: 'int', default: 3 },
      { key: 'batch_size', label: 'Batch Size', type: 'int', default: 16 },
      { key: 'learning_rate', label: 'Learning Rate', type: 'float', default: 0.00002 },
    ],
  };
};

// ── Chat ─────────────────────────────────────────────────────────

export const mockPins = [
  {
    id: 'pin-001',
    title: 'GPU Utilization (7d)',
    query: 'Show GPU utilization over the last 7 days',
    widget: {
      type: 'line',
      title: 'GPU Utilization %',
      xAxisKey: 'time',
      series: [{ name: 'GPU Util', dataKey: 'value', color: '#8b5cf6' }],
      data: Array.from({ length: 7 }, (_, i) => ({
        time: `Day ${i + 1}`,
        value: 45 + Math.round(Math.random() * 40),
      })),
    },
    created_at: ago(48),
    user: 'xiaohan',
  },
  {
    id: 'pin-002',
    title: 'Cost by Workflow',
    query: 'Show cost breakdown by workflow',
    widget: {
      type: 'pie',
      title: 'Cost by Workflow',
      data: [
        { name: 'text2sql', value: 45 },
        { name: 'llm-sft', value: 120 },
        { name: 'reward-model', value: 192 },
        { name: 'spark-etl', value: 12 },
      ],
    },
    created_at: ago(24),
    user: 'xiaohan',
  },
];

// ── Settings ─────────────────────────────────────────────────────

export const mockSettings = {
  theme: 'dark',
  default_namespace: 'ml-platform',
  default_gpu_type: 'A100',
  default_recipe_profile: 'Medium',
  editor_font_size: 14,
  editor_key_bindings: 'Default',
  editor_minimap: true,
  editor_word_wrap: true,
  burn_rate_display: 'Badge',
  budget_alerts_enabled: true,
  weekly_report_email: true,
  budget_limit_monthly: 50000,
  idle_timeout_minutes: 120,
  auto_suspend_action: 'Suspend',
  ghost_desk_protection: true,
  reduce_motion: false,
  high_contrast: false,
  notification_job_complete: true,
  notification_idle_gpu: true,
  notification_budget_threshold: true,
  notification_budget_percent: 80,
};

// ── Auth ─────────────────────────────────────────────────────────

export const mockAuth = {
  authenticated: true,
  user: {
    sub: 'mock-user-001',
    name: 'Dev User',
    email: 'dev@ml-platform.local',
    role: 'admin',
    groups: ['ml-engineers', 'platform-admins'],
  },
};

// ── Config ───────────────────────────────────────────────────────

export const mockConfig = {
  grafanaUrl: 'http://localhost:3000',
};
