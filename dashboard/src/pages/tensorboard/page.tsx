import { useQuery } from '@tanstack/react-query';
import {
  ExternalLink, RefreshCw, BarChart3, Loader2, AlertTriangle,
} from 'lucide-react';
import { api } from '@/lib/api';

interface TBRun {
  execution_id: string;
  s3_path: string;
}

export default function TensorBoardPage() {
  const {
    data: runsData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['tensorboard-runs'],
    queryFn: () => api.tensorboard.getRuns(),
    staleTime: 30_000,
  });

  const { data: tbBaseUrl } = useQuery({
    queryKey: ['tensorboard-url'],
    queryFn: () => api.tensorboard.getUrl(),
    staleTime: 60_000,
  });

  const runs: TBRun[] = runsData?.runs ?? [];

  const openTensorBoard = (executionId?: string) => {
    if (!tbBaseUrl) return;
    const url = executionId
      ? `${tbBaseUrl}/#scalars&regexInput=${encodeURIComponent(executionId)}`
      : tbBaseUrl;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div style={{ padding: 24, maxWidth: 960 }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 24,
      }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0 }}>TensorBoard</h1>
          <p style={{ fontSize: 13, color: 'var(--text-dimmed)', margin: '4px 0 0' }}>
            View training metrics, scalars, and histograms for all runs.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => refetch()}
            title="Refresh runs"
          >
            <RefreshCw style={{ width: 14, height: 14 }} /> Refresh
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => openTensorBoard()}
            disabled={!tbBaseUrl}
          >
            <ExternalLink style={{ width: 14, height: 14 }} /> Open Full TensorBoard
          </button>
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: 48, color: 'var(--text-dimmed)',
        }}>
          <style>{'@keyframes spin { to { transform: rotate(360deg); } }'}</style>
          <Loader2 style={{ width: 20, height: 20, animation: 'spin 1s linear infinite', marginRight: 8 }} />
          Loading TensorBoard runs...
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: 16, background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8,
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 13, color: 'var(--error)',
        }}>
          <AlertTriangle style={{ width: 14, height: 14 }} />
          Failed to load TensorBoard runs. Is the backend running?
        </div>
      )}

      {/* Runs list */}
      {!isLoading && !error && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {runs.length === 0 ? (
            <div style={{
              padding: 48, textAlign: 'center', color: 'var(--text-dimmed)',
              background: 'rgba(255,255,255,0.02)', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <BarChart3 style={{ width: 32, height: 32, margin: '0 auto 12px', opacity: 0.4 }} />
              <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>No TensorBoard logs yet</div>
              <div style={{ fontSize: 12 }}>
                Use <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,0.06)', padding: '1px 6px', borderRadius: 3 }}>
                  get_summary_writer()
                </code> from the SDK to start logging metrics.
              </div>
            </div>
          ) : (
            runs.map((run) => (
              <div
                key={run.execution_id}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '12px 16px', borderRadius: 8,
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  transition: 'border-color 0.15s ease',
                }}
                onMouseOver={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.3)'; }}
                onMouseOut={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'; }}
              >
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500 }}>{run.execution_id}</div>
                  <div style={{
                    fontSize: 11, color: 'var(--text-dimmed)',
                    fontFamily: 'var(--font-mono)', marginTop: 2,
                  }}>
                    {run.s3_path}
                  </div>
                </div>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => openTensorBoard(run.execution_id)}
                  disabled={!tbBaseUrl}
                >
                  <ExternalLink style={{ width: 12, height: 12 }} /> Open
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
