
import { useState, useRef, useEffect } from 'react';
import { Send, MessageCircle, Pin } from 'lucide-react';
import { apiClient } from '@/lib/api';
import WidgetRenderer, { WidgetSpec } from '@/components/WidgetRenderer';

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
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (message?: string) => {
    const msg = message || input.trim();
    if (!msg || isLoading) return;

    setInput('');
    setError('');
    const userMsg: Message = { role: 'user', content: msg };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const { data } = await apiClient.post('/chat/ask', { message: msg, history });
      
      let content = data.response || data.answer || data.text || '';
      let widget = data.widget;

      // Sometimes the backend sends the entire object as a stringified JSON inside `response` or `text`
      if (typeof content === 'string') {
        try {
          const parsed = JSON.parse(content);
          if (parsed.text || parsed.widget) {
            content = parsed.text || parsed.response || parsed.answer || '';
            widget = parsed.widget;
          }
        } catch(e) {
          // not JSON, leave as is
        }
      }

      // Handle raw string responses that might contain JSON
      if (!content && typeof data === 'string') {
        try {
          const parsed = JSON.parse(data);
          content = parsed.response || parsed.answer || parsed.text || '';
          widget = parsed.widget;
        } catch(e) {
          content = data;
        }
      } else if (!content) {
        content = JSON.stringify(data);
      }

      setMessages((prev) => [...prev, { role: 'assistant', content, widget }]);
    } catch (err: any) {
      setError(err.message || 'Failed to get response');
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Sorry, I couldn\'t process that request. The AI assistant backend may not be configured yet.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-container" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 32px)' }}>
      <div className="page-header" style={{ flexShrink: 0 }}>
        <h1>AI Assistant</h1>
        <p>Ask questions about your cluster, jobs, metrics, and costs</p>
      </div>

      {/* Messages area */}
      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
          {messages.length === 0 && !isLoading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center' }}>
              <MessageCircle style={{ width: 48, height: 48, color: 'var(--accent-primary)', opacity: 0.4, marginBottom: 16 }} />
              <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8, color: 'var(--text-primary)' }}>What would you like to know?</h2>
              <p style={{ fontSize: 13, color: 'var(--text-dimmed)', marginBottom: 24, maxWidth: 420 }}>
                Ask about GPU utilization, training jobs, cluster costs, or anything else about your ML infrastructure.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, maxWidth: 500 }}>
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    style={{
                      textAlign: 'left',
                      fontSize: 13,
                      padding: '10px 14px',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid rgba(255,255,255,0.08)',
                      background: 'rgba(255,255,255,0.03)',
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(124,58,237,0.3)';
                      e.currentTarget.style.background = 'rgba(124,58,237,0.06)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                      e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {messages.map((msg, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <div style={{
                  maxWidth: '70%',
                  padding: '12px 16px',
                  borderRadius: 12,
                  fontSize: 14,
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap',
                  ...(msg.role === 'user'
                    ? {
                        background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                        color: '#fff',
                      }
                    : {
                        background: 'rgba(255,255,255,0.04)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        color: 'var(--text-primary)',
                      }),
                }}>
                  {msg.content}
                </div>
                {msg.widget && (
                  <div style={{ marginTop: 12, width: '100%', maxWidth: '800px', alignSelf: 'flex-start' }}>
                    <WidgetRenderer spec={msg.widget} className="!bg-[var(--card-bg)]" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Loading indicator */}
          {isLoading && (
            <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: 16 }}>
              <div style={{
                padding: '12px 16px',
                borderRadius: 12,
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-primary)', animation: 'pulse 1.4s infinite' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-primary)', animation: 'pulse 1.4s infinite 0.2s' }} />
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-primary)', animation: 'pulse 1.4s infinite 0.4s' }} />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid rgba(255,255,255,0.06)',
          flexShrink: 0,
        }}>
          <form
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            style={{ display: 'flex', gap: 10 }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your ML infrastructure..."
              disabled={isLoading}
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(255,255,255,0.04)',
                color: 'var(--text-primary)',
                fontSize: 14,
                outline: 'none',
              }}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="btn btn-primary"
              style={{ display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <Send style={{ width: 14, height: 14 }} />
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
