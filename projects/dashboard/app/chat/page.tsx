'use client';

import { useState, useRef, useEffect } from 'react';
import { useChatStream } from '@/lib/useSSE';
import WidgetRenderer, { WidgetSpec } from '@/components/WidgetRenderer';
import { apiClient } from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  widget?: WidgetSpec;
}

const EXAMPLE_QUESTIONS = [
  'How many nodes are in the cluster?',
  'Show me GPU utilization for the last hour',
  'List all running pods with GPUs',
  'Show all MLflow experiments',
  'What is the total cost this week?',
  'What Ray jobs are running?',
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { text, widget, isStreaming, error, send } = useChatStream();

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, text]);

  // When streaming completes, add the assistant message
  useEffect(() => {
    if (!isStreaming && (text || widget)) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: text, widget: widget || undefined },
      ]);
    }
  }, [isStreaming]);

  const handleSend = async (message?: string) => {
    const msg = message || input.trim();
    if (!msg || isStreaming) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: msg }]);

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    send(msg, history);
  };

  const handlePin = async (spec: WidgetSpec) => {
    try {
      await apiClient.post('/chat/pins', {
        title: spec.title,
        query: messages[messages.length - 2]?.content || spec.title,
        widget: spec,
      });
    } catch {
      // silently fail — pin is best-effort
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] p-4">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-900">AI Assistant</h1>
        <p className="text-gray-500 text-sm">
          Ask questions about your cluster, jobs, metrics, and costs
        </p>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto space-y-4 pb-4">
        {messages.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-4xl mb-4">💬</div>
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              What would you like to know?
            </h2>
            <p className="text-gray-500 text-sm mb-6 max-w-md">
              Ask about GPU utilization, training jobs, cluster costs, or anything else
              about your ML infrastructure.
            </p>
            <div className="grid grid-cols-2 gap-2 max-w-lg">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="text-left text-sm px-3 py-2 rounded-lg border border-gray-200
                    hover:border-blue-300 hover:bg-blue-50 text-gray-600 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-2xl rounded-lg px-4 py-2 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {msg.content && <p className="whitespace-pre-wrap">{msg.content}</p>}
              {msg.widget && (
                <div className="mt-3">
                  <WidgetRenderer spec={msg.widget} onPin={handlePin} />
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-2xl rounded-lg px-4 py-2 bg-gray-100 text-gray-900">
              {text && <p className="whitespace-pre-wrap">{text}</p>}
              {widget && (
                <div className="mt-3">
                  <WidgetRenderer spec={widget} onPin={handlePin} />
                </div>
              )}
              {!text && !widget && (
                <div className="flex items-center gap-2 text-gray-400">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:0.1s]" />
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:0.2s]" />
                </div>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-start">
            <div className="rounded-lg px-4 py-2 bg-red-50 text-red-600 text-sm">
              {error}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 pt-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your ML infrastructure..."
            disabled={isStreaming}
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
              disabled:bg-gray-50 disabled:text-gray-400"
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white
              hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
              transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
