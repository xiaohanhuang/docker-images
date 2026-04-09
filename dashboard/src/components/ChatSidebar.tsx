import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { useChatStream } from '@/lib/useSSE';
import WidgetRenderer, { WidgetSpec } from '@/components/WidgetRenderer';
import { api } from '@/lib/api';
import { useChat } from '@/lib/ChatContext';
import { X, MessageCircle } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  widget?: WidgetSpec;
}

export function ChatSidebar() {
  const { isOpen, closeChat, initialMessage } = useChat();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { text, widget, isStreaming, error, send } = useChatStream();

  // Handle initial message when sidebar opens
  useEffect(() => {
    if (isOpen && initialMessage) {
      handleSend(initialMessage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, initialMessage]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, text, widget]);

  // Track previous streaming state to detect completion
  const wasStreamingRef = useRef(false);

  // Handle stream completion
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming && (text || widget)) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: text, widget: widget || undefined },
      ]);
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming, text, widget]);

  const handleSend = useCallback(async (message?: string) => {
    const msg = message || input.trim();
    if (!msg || isStreaming) return;

    setInput('');
    const nextMessages: Message[] = [...messages, { role: 'user', content: msg }];
    setMessages(nextMessages);

    const history = nextMessages.map((m) => ({ role: m.role, content: m.content }));
    send(msg, history);
  }, [input, isStreaming, messages, send]);

  const handlePin = async (spec: WidgetSpec) => {
    try {
      await api.chat.pin({
        title: spec.title,
        query: messages[messages.length - 2]?.content || spec.title,
        widget: spec as any,
      });
    } catch {
      // ignore
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-[450px] bg-white shadow-2xl z-50 flex flex-col border-l border-gray-200 animate-in slide-in-from-right duration-300">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between bg-gray-50">
        <div className="flex items-center gap-2">
          <MessageCircle className="w-5 h-5 text-blue-600" />
          <h2 className="font-bold text-gray-900">AI Assistant</h2>
        </div>
        <button
          onClick={closeChat}
          className="p-1 hover:bg-gray-200 rounded-full transition-colors text-gray-500"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[90%] rounded-lg px-3 py-2 text-sm ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {msg.content && (
                msg.role === 'user'
                  ? <p className="whitespace-pre-wrap">{msg.content}</p>
                  : <div className="prose prose-sm prose-gray max-w-none [&_pre]:bg-gray-800 [&_pre]:text-gray-100 [&_pre]:rounded [&_pre]:p-2 [&_pre]:overflow-x-auto [&_code]:text-xs [&_h2]:text-base [&_h2]:font-bold [&_h2]:mt-3 [&_h2]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_p]:my-1">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
              )}
              {msg.widget && (
                <div className="mt-2 scale-95 origin-top-left">
                  <WidgetRenderer spec={msg.widget} onPin={handlePin} />
                </div>
              )}
            </div>
          </div>
        ))}

        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-[90%] rounded-lg px-3 py-2 bg-gray-100 text-gray-900 text-sm">
              {text && (
                <div className="prose prose-sm prose-gray max-w-none [&_pre]:bg-gray-800 [&_pre]:text-gray-100 [&_pre]:rounded [&_pre]:p-2 [&_pre]:overflow-x-auto [&_code]:text-xs [&_h2]:text-base [&_h2]:font-bold [&_h2]:mt-3 [&_h2]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5 [&_p]:my-1">
                  <ReactMarkdown>{text}</ReactMarkdown>
                </div>
              )}
              {widget && (
                <div className="mt-2 scale-95 origin-top-left">
                  <WidgetRenderer spec={widget} onPin={handlePin} />
                </div>
              )}
              {!text && !widget && (
                <div className="flex items-center gap-1.5 py-1">
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" />
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:0.1s]" />
                  <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:0.2s]" />
                </div>
              )}
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-start">
            <div className="max-w-[90%] rounded-lg px-3 py-2 bg-red-50 text-red-700 text-sm border border-red-200">
              <p>Failed to get a response. Please try again.</p>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
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
            placeholder="Ask about your infrastructure..."
            disabled={isStreaming}
            className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
              disabled:bg-gray-50 disabled:text-gray-400"
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white
              hover:bg-blue-700 disabled:bg-gray-300 transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
