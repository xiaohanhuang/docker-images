import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const IS_MOCK = process.env.VITE_MOCK === 'true';

/**
 * Vite dev-server plugin: intercept /api/* requests and return
 * fixture data so the dashboard works without a running backend.
 * Only loaded when VITE_MOCK=true.
 */
function mockApiPlugin(): Plugin {
  return {
    name: 'mock-api',
    configureServer(server) {
      server.middlewares.use('/api', async (req, res) => {
        // Dynamically import mock data from src (runs in Node via Vite SSR)
        const mod = await server.ssrLoadModule('/src/lib/mock-data.ts');

        const url = req.url || '/';
        const route = url.replace(/^\//, '').replace(/^v1\//, '').split('?')[0]; // strip leading slash & v1/ & query

        const json = (data: unknown, status = 200) => {
          res.writeHead(status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(data));
        };

        // ---------- GET routes ----------
        if (req.method === 'GET') {
          if (route === 'auth') return json({ authenticated: true, user: { sub: 'xiaohan', name: 'Xiaohan Huang', email: 'xiaohan@ml-platform.local', role: 'admin', groups: ['ml-engineers'] } });
          if (route === 'config') return json({ grafanaUrl: 'http://grafana.mock.local' });
          if (route === 'dashboard/overview') return json(mod.mockDashboardOverview);
          if (route === 'jobs') return json({ jobs: mod.mockJobs });
          if (route.match(/^jobs\/[^/]+\/logs$/)) { res.writeHead(200, { 'Content-Type': 'text/plain' }); return res.end(mod.mockJobLogs); }
          if (route.startsWith('jobs/')) return json(mod.mockJobs.find((j: any) => j.job_id === route.replace('jobs/', '')) || mod.mockJobs[0]);
          if (route === 'desks') return json(mod.mockDesks);
          if (route === 'cost/report') return json(mod.mockCostReport);
          if (route === 'pods') return json({ pods: mod.mockPods });
          if (route === 'mlflow/experiments') return json(mod.mockExperiments);
          if (route.match(/^mlflow\/experiments\/[^/]+\/runs$/)) return json(mod.mockRuns);
          if (route === 'mlflow/models') return json(mod.mockModels);
          if (route.startsWith('mlflow/models/')) return json(mod.mockModelDetail(decodeURIComponent(route.replace('mlflow/models/', ''))));
          if (route === 'kubernetes/nodes') return json(mod.mockNodes);
          if (route === 'kubernetes/events') return json(mod.mockEvents);
          if (route === 'ray/cluster') return json(mod.mockRayCluster);
          if (route === 'ray/jobs') return json(mod.mockRayJobs);
          if (route === 'serving') return json(mod.mockServing);
          if (route === 'recipes') return json(mod.mockRecipes);
          if (route.startsWith('recipes/')) return json(mod.mockRecipeDetail(decodeURIComponent(route.replace('recipes/', ''))));
          if (route === 'chat/pins') return json(mod.mockPins);
          if (route === 'settings') return json(mod.mockSettings);
          if (route === 'components') return json({ components: [
            { name: 'data-loader', version: '1.0.0', desc: 'Load data from S3/GCS', type: 'task' },
            { name: 'tokenizer', version: '2.1.0', desc: 'HuggingFace tokenizers', type: 'task' },
            { name: 'gpu-trainer', version: '1.3.0', desc: 'Distributed GPU training', type: 'task' },
            { name: 'sft-pipeline', version: '1.2.0', desc: 'End-to-end SFT workflow', type: 'workflow' },
          ]});
        }

        // ---------- POST routes ----------
        if (req.method === 'POST') {
          // Collect body
          const chunks: Buffer[] = [];
          for await (const chunk of req) chunks.push(chunk as Buffer);
          const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};

          if (route === 'chat/ask') return json({ text: `[Mock] Answer for: "${body.message || 'question'}"`, widget: { type: 'stat', title: 'Mock', value: '42', unit: 'items', trend: 12.5 } });
          if (route === 'desks') return json({ status: 'launched', desk_id: `desk-${Date.now()}`, message: `Desk ${body.name} created (mock)` });
          if (route.match(/^desks\/[^/]+\/run$/)) return json({ stdout: `[Mock] Executed:\n${body.code || ''}\n>>> OK`, stderr: '', images: [] });
          if (route === 'chat/pins') return json({ id: `pin-${Date.now()}`, ...body, created_at: new Date().toISOString() });
          if (route === 'serving') return json({ status: 'deployed', name: body.name, message: 'Deployed (mock)' });
          if (route === 'jobs') return json({ job_id: `f-mock-${Date.now()}`, status: 'PENDING', message: 'Submitted (mock)' });
        }

        // ---------- DELETE routes ----------
        if (req.method === 'DELETE') {
          if (route.startsWith('desks/')) return json({ status: 'stopped', desk_id: route.replace('desks/', '') });
          if (route.startsWith('chat/pins/')) return json({ status: 'unpinned' });
        }

        // ---------- PUT/PATCH routes ----------
        if (req.method === 'PUT' || req.method === 'PATCH') {
          if (route === 'settings') {
            const chunks: Buffer[] = [];
            for await (const chunk of req) chunks.push(chunk as Buffer);
            const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
            return json({ ...mod.mockSettings, ...body });
          }
        }

        // Fallback
        return json({ error: `Mock route not found: /api/${route}` }, 404);
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), ...(IS_MOCK ? [mockApiPlugin()] : [])],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: process.env.VITE_MOCK === 'true'
      ? undefined
      : {
          // Domain API routes — backend serves them at /api/v1/*
          '/api/v1': {
            target: process.env.VITE_API_URL || 'http://localhost:8000',
            changeOrigin: true,
            ws: true,
          },
          // Root-level routes (auth, config, health) — strip /api prefix
          '/api/auth': {
            target: process.env.VITE_API_URL || 'http://localhost:8000',
            changeOrigin: true,
            rewrite: () => '/auth/me',
          },
          '/api/config': {
            target: process.env.VITE_API_URL || 'http://localhost:8000',
            changeOrigin: true,
            rewrite: (p) => p.replace(/^\/api/, ''),
          },
          '/proxy': {
            target: 'http://localhost:8001',
            changeOrigin: true,
          },
          '/desk-proxy': {
            target: 'http://localhost:8000',
            changeOrigin: true,
            ws: true,
          },
          '/desk-marimo': {
            target: 'http://localhost:8000',
            changeOrigin: true,
            ws: true,
          },
          '/desk-jupyter': {
            target: 'http://localhost:8000',
            changeOrigin: true,
            ws: true,
          },
        },
  },
});
