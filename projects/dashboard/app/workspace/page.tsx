'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import WidgetRenderer from '@/components/WidgetRenderer';

interface PinnedWidget {
  id: string;
  title: string;
  query: string;
  widget: Record<string, unknown>;
  created_at: string;
}

export default function WorkspacePage() {
  const queryClient = useQueryClient();

  const { data: pins = [], isLoading } = useQuery<PinnedWidget[]>({
    queryKey: ['pins'],
    queryFn: api.chat.listPins,
    refetchInterval: 30_000,
  });

  const unpinMutation = useMutation({
    mutationFn: (pinId: string) => api.chat.unpin(pinId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['pins'] }),
  });

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Workspace</h1>
        <p className="text-gray-500 text-sm">
          Your pinned widgets and saved queries
        </p>
      </div>

      {isLoading && (
        <div className="text-gray-400 text-sm">Loading pinned widgets...</div>
      )}

      {!isLoading && pins.length === 0 && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-lg mb-2">No pinned widgets yet</p>
          <p className="text-sm">
            Ask the AI assistant a question and pin useful charts to your workspace.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {pins.map((pin) => (
          <div key={pin.id} className="bg-white rounded-lg shadow border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400">
                Pinned {new Date(pin.created_at).toLocaleDateString()}
              </span>
              <button
                onClick={() => unpinMutation.mutate(pin.id)}
                className="text-xs text-red-500 hover:text-red-700"
              >
                Unpin
              </button>
            </div>
            <p className="text-sm text-gray-500 mb-3 italic">&ldquo;{pin.query}&rdquo;</p>
            <WidgetRenderer spec={pin.widget as never} />
          </div>
        ))}
      </div>
    </div>
  );
}
