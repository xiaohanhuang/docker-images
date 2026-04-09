/**
 * useSSE — React hook for consuming Server-Sent Events from the chat live endpoint.
 *
 * Used to power real-time metric updates in live widgets.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import { IS_MOCK } from './api';

// Always use the relative /api path — the Next.js rewrite in next.config.js
// proxies these requests to the backend service at runtime.
const API_BASE_URL = '/api';

interface SSEOptions {
  /** PromQL query for the /chat/live endpoint */
  query: string;
  /** Whether to connect (set false to pause) */
  enabled?: boolean;
}

interface SSEState<T> {
  data: T | null;
  error: string | null;
  isConnected: boolean;
}

export function useSSE<T = any>({ query, enabled = true }: SSEOptions): SSEState<T> {
  const [state, setState] = useState<SSEState<T>>({
    data: null,
    error: null,
    isConnected: false,
  });
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!enabled || !query) {
      return;
    }

    // ── Mock mode: emit a synthetic metric every 2 s ──
    if (IS_MOCK) {
      setState((prev) => ({ ...prev, isConnected: true, error: null }));
      const interval = setInterval(() => {
        setState((prev) => ({
          ...prev,
          data: { value: 40 + Math.random() * 50, timestamp: Date.now() } as T,
        }));
      }, 2000);
      return () => {
        clearInterval(interval);
        setState((prev) => ({ ...prev, isConnected: false }));
      };
    }

    const url = `${API_BASE_URL}/chat/live?query=${encodeURIComponent(query)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      setState((prev) => ({ ...prev, isConnected: true, error: null }));
    };

    es.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'metric') {
          setState((prev) => ({ ...prev, data: parsed.content }));
        } else if (parsed.type === 'error') {
          setState((prev) => ({ ...prev, error: parsed.content }));
        }
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      setState((prev) => ({ ...prev, isConnected: false, error: 'Connection lost' }));
      es.close();
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
      setState((prev) => ({ ...prev, isConnected: false }));
    };
  }, [query, enabled]);

  return state;
}


/**
 * useChatSSE — stream a chat response via SSE (POST /chat/).
 *
 * Returns text chunks and a final widget as they arrive.
 */
interface ChatSSEState {
  text: string;
  widget: any | null;
  isStreaming: boolean;
  error: string | null;
}

export function useChatStream() {
  const [state, setState] = useState<ChatSSEState>({
    text: '',
    widget: null,
    isStreaming: false,
    error: null,
  });

  const send = useCallback(async (message: string, history: { role: string; content: string }[]) => {
    setState({ text: '', widget: null, isStreaming: true, error: null });

    // ── Mock mode: simulate a streaming response ──
    if (IS_MOCK) {
      const isAnalyze = message.toLowerCase().startsWith('analyze job');
      const mockText = isAnalyze
        ? `## Post-Training Autopsy

**Job:** ${message.replace(/analyze job\s*/i, '')}
**Status:** SUCCEEDED | **Duration:** 2h 14m | **GPU Utilization:** 62.3%

### Findings

1. **Low GPU utilization (62.3%)** — The average GPU utilization across 4× A100 workers was well below the 80% target. The training loop is likely bottlenecked by data loading or CPU preprocessing, leaving GPUs idle between steps.

2. **No OOM errors detected** — Logs show no "Out of Memory" or "CUDA error" events. Current batch size (per_device_train_batch_size=4) is conservative for A100 80GB — there is significant headroom.

3. **Checkpoint I/O spikes** — Every 500 steps, GPU utilization drops to ~5% for 45 seconds during checkpoint writes to EFS. This accounts for ~12% of total wall-clock time.

### Recommendations

- **Increase batch size from 4 → 16** with gradient_accumulation_steps=2. This will improve GPU utilization to ~85% and reduce total training time by ~35%. Memory usage will grow from 28GB → 58GB per GPU, still within A100 80GB limits.

- **Enable async checkpointing** — Use \`torch.distributed.checkpoint.async_save()\` or write checkpoints to a background thread. This eliminates the 45s GPU stall per checkpoint and saves ~12% wall-clock time.

- **Add 4 DataLoader workers** (num_workers=4, pin_memory=True, prefetch_factor=4). CPU preprocessing is the current bottleneck. This will keep the GPU pipeline saturated between steps.`
        : `Here's a summary based on your query: "${message}". The platform is running 3 active pods with 13 GPUs allocated across 4 nodes. Total cost this week is $217.20.`;

      const words = mockText.split(' ');
      for (const word of words) {
        await new Promise((r) => setTimeout(r, 30));
        setState((prev) => ({ ...prev, text: prev.text + word + ' ' }));
      }

      const mockWidget = isAnalyze
        ? {
            type: 'table' as const,
            title: 'Training Metrics',
            columns: ['Metric', 'Current', 'Recommended'],
            rows: [
              ['Batch Size', '4', '16'],
              ['GPU Utilization', '62.3%', '~85%'],
              ['Checkpoint Time', '45s (sync)', '< 2s (async)'],
              ['DataLoader Workers', '0', '4'],
              ['Est. Training Time', '2h 14m', '~1h 20m'],
            ],
          }
        : {
            type: 'stat' as const,
            title: 'Active GPUs',
            value: '13',
            unit: 'GPUs',
            trend: 8.3,
          };

      setState((prev) => ({
        ...prev,
        isStreaming: false,
        widget: mockWidget,
      }));
      return;
    }

    try {
      const resp = await fetch(`${API_BASE_URL}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User': 'dashboard',
        },
        body: JSON.stringify({ message, history }),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const reader = resp.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const event = JSON.parse(trimmed.slice(6));
            if (event.type === 'text') {
              setState((prev) => ({ ...prev, text: prev.text + event.content }));
            } else if (event.type === 'widget') {
              setState((prev) => ({ ...prev, widget: event.content }));
            } else if (event.type === 'done') {
              setState((prev) => ({ ...prev, isStreaming: false }));
            }
          } catch {
            // skip malformed
          }
        }
      }

      setState((prev) => ({ ...prev, isStreaming: false }));
    } catch (e: any) {
      setState((prev) => ({
        ...prev,
        isStreaming: false,
        error: e.message || 'Stream failed',
      }));
    }
  }, []);

  return { ...state, send };
}
