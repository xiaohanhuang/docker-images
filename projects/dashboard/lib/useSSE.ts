/**
 * useSSE — React hook for consuming Server-Sent Events from the chat live endpoint.
 *
 * Used to power real-time metric updates in live widgets.
 */
import { useEffect, useRef, useState, useCallback } from 'react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

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
