
import { useState } from 'react';
import { GitMerge, Database, ArrowRight, FileText, Clock, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

/* ──── enhanced lineage graph data ──── */
const NODES = [
  // Datasets
  { id: 'ds-guanaco', label: 'openassistant-guanaco', type: 'dataset', x: 50, y: 60, version: 'v3.1', size: '2.4 GB', records: '189K' },
  { id: 'ds-reward', label: 'rlhf-reward-pairs', type: 'dataset', x: 50, y: 160, version: 'v2.0', size: '890 MB', records: '412K' },
  { id: 'ds-code', label: 'code-instruct-252k', type: 'dataset', x: 50, y: 260, version: 'v1.3', size: '1.1 GB', records: '252K' },
  { id: 'ds-spider', label: 'text2sql-spider', type: 'dataset', x: 50, y: 360, version: 'v4.0', size: '340 MB', records: '78K' },
  { id: 'ds-mmlu', label: 'eval-mmlu-benchmark', type: 'dataset', x: 50, y: 460, version: 'v1.0', size: '120 MB', records: '14K' },
  // Pipelines
  { id: 'pipe-sft', label: 'llm-sft-training', type: 'pipeline', x: 380, y: 100, status: 'completed' },
  { id: 'pipe-reward', label: 'reward-model-train', type: 'pipeline', x: 380, y: 200, status: 'completed' },
  { id: 'pipe-rlhf', label: 'rlhf-ppo-alignment', type: 'pipeline', x: 380, y: 300, status: 'running' },
  { id: 'pipe-t2sql', label: 'text2sql-finetune', type: 'pipeline', x: 380, y: 400, status: 'completed' },
  // Models
  { id: 'model-sft', label: 'llama-2-70b-sft', type: 'model', x: 680, y: 100, version: 'v3', stage: 'Staging' },
  { id: 'model-reward', label: 'reward-model-v2', type: 'model', x: 680, y: 200, version: 'v2', stage: 'Production' },
  { id: 'model-rlhf', label: 'llama-2-70b-rlhf', type: 'model', x: 680, y: 300, version: 'v4', stage: 'Production' },
  { id: 'model-t2sql', label: 'text2sql-7b', type: 'model', x: 680, y: 400, version: 'v2', stage: 'Staging' },
  // Endpoints
  { id: 'ep-prod', label: 'llm-prod-v3', type: 'endpoint', x: 920, y: 300, traffic: '85%' },
];

const EDGES = [
  { from: 'ds-guanaco', to: 'pipe-sft', type: 'Train' },
  { from: 'ds-mmlu', to: 'pipe-sft', type: 'Eval' },
  { from: 'ds-reward', to: 'pipe-reward', type: 'Train' },
  { from: 'ds-code', to: 'pipe-rlhf', type: 'Train' },
  { from: 'ds-spider', to: 'pipe-t2sql', type: 'Train' },
  { from: 'pipe-sft', to: 'model-sft', type: 'Output' },
  { from: 'pipe-reward', to: 'model-reward', type: 'Output' },
  { from: 'model-sft', to: 'pipe-rlhf', type: 'Input' },
  { from: 'model-reward', to: 'pipe-rlhf', type: 'Input' },
  { from: 'pipe-rlhf', to: 'model-rlhf', type: 'Output' },
  { from: 'pipe-t2sql', to: 'model-t2sql', type: 'Output' },
  { from: 'model-rlhf', to: 'ep-prod', type: 'Serving' },
];

/* Datasets derived from the lineage graph — enriched with catalog info */
const datasets = NODES.filter(n => n.type === 'dataset').map(n => ({
  name: n.label,
  version: (n as any).version,
  format: n.label.includes('sql') ? 'Parquet' : n.label.includes('eval') ? 'JSON' : 'Parquet',
  size: (n as any).size,
  records: (n as any).records,
  source: n.label.includes('guanaco') ? 'HuggingFace'
    : n.label.includes('reward') ? 'Internal'
    : n.label.includes('code') ? 'HuggingFace'
    : n.label.includes('spider') ? 'GitHub'
    : 'HuggingFace',
  updated: n.label.includes('guanaco') ? '3d ago'
    : n.label.includes('reward') ? '1w ago'
    : n.label.includes('code') ? '2w ago'
    : n.label.includes('spider') ? '5d ago'
    : '1d ago',
}));

function getNodeColor(type: string) {
  switch (type) {
    case 'dataset': return '#06b6d4';
    case 'pipeline': return '#7c3aed';
    case 'model': return '#10b981';
    case 'endpoint': return '#f59e0b';
    default: return '#94a3b8';
  }
}

function NodeIcon({ type }: { type: string }) {
  const w = 14;
  switch (type) {
    case 'dataset': return <Database style={{ width: w, height: w, color: getNodeColor(type) }} />;
    case 'pipeline': return <GitMerge style={{ width: w, height: w, color: getNodeColor(type) }} />;
    case 'model': return <FileText style={{ width: w, height: w, color: getNodeColor(type) }} />;
    case 'endpoint': return <ArrowRight style={{ width: w, height: w, color: getNodeColor(type) }} />;
    default: return null;
  }
}

export default function DataPage() {
  const [selected, setSelected] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const selectedNode = NODES.find(n => n.id === selected);

  // Get upstream/downstream of selection
  const upstreamIds = selected ? EDGES.filter(e => e.to === selected).map(e => e.from) : [];
  const downstreamIds = selected ? EDGES.filter(e => e.from === selected).map(e => e.to) : [];
  const highlightedIds = selected ? new Set([selected, ...upstreamIds, ...downstreamIds]) : null;
  const highlightedEdges = selected ? new Set(
    EDGES.filter(e => (e.from === selected || e.to === selected)).map(e => `${e.from}-${e.to}`)
  ) : null;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Data Lineage & Datasets</h1>
        <p>Interactive bidirectional data lineage graph — click any node to trace upstream and downstream</p>
      </div>

      {/* Interactive Lineage Graph */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title">Lineage Graph</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {/* Legend */}
            {['dataset', 'pipeline', 'model', 'endpoint'].map(type => (
              <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: getNodeColor(type) }} />
                <span style={{ color: 'var(--text-dimmed)', textTransform: 'capitalize' }}>{type}</span>
              </div>
            ))}
            <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.1)', margin: '0 4px' }} />
            <button className="btn btn-ghost btn-sm" onClick={() => setZoom(z => Math.min(1.5, z + 0.1))}>
              <ZoomIn style={{ width: 12, height: 12 }} />
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}>
              <ZoomOut style={{ width: 12, height: 12 }} />
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => { setZoom(1); setSelected(null); }}>
              <Maximize2 style={{ width: 12, height: 12 }} />
            </button>
          </div>
        </div>
        <div className="card-body" style={{ overflow: 'auto', position: 'relative' }}>
          <svg
            width={1060 * zoom} height={530 * zoom}
            viewBox="0 0 1060 530"
            style={{ display: 'block', margin: '0 auto' }}
          >
            <defs>
              <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 3 L 0 6 z" fill="rgba(255,255,255,0.2)" />
              </marker>
              <marker id="arrow-hl" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 3 L 0 6 z" fill="var(--accent-primary)" />
              </marker>
            </defs>

            {/* Draw edges */}
            {EDGES.map(edge => {
              const from = NODES.find(n => n.id === edge.from)!;
              const to = NODES.find(n => n.id === edge.to)!;
              const hl = highlightedEdges?.has(`${edge.from}-${edge.to}`);
              const dimmed = highlightedEdges && !hl;
              const x1 = from.x + 140;
              const y1 = from.y + 20;
              const x2 = to.x;
              const y2 = to.y + 20;
              const mx = (x1 + x2) / 2;
              return (
                <path key={`${edge.from}-${edge.to}`}
                  d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                  fill="none"
                  stroke={hl ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)'}
                  strokeWidth={hl ? 2 : 1}
                  strokeOpacity={dimmed ? 0.15 : 1}
                  markerEnd={hl ? 'url(#arrow-hl)' : 'url(#arrow)'}
                  style={{ transition: 'all 0.3s ease' }}
                />
              );
            })}

            {/* Draw nodes */}
            {NODES.map(node => {
              const isHighlighted = !highlightedIds || highlightedIds.has(node.id);
              const isSelected = selected === node.id;
              const color = getNodeColor(node.type);
              return (
                <g key={node.id}
                  onClick={() => setSelected(s => s === node.id ? null : node.id)}
                  style={{ cursor: 'pointer', transition: 'opacity 0.3s ease', opacity: isHighlighted ? 1 : 0.2 }}
                >
                  <rect
                    x={node.x} y={node.y} width={140} height={40} rx={6}
                    fill={isSelected ? `${color}15` : 'rgba(255,255,255,0.03)'}
                    stroke={isSelected ? color : 'rgba(255,255,255,0.08)'}
                    strokeWidth={isSelected ? 2 : 1}
                  />
                  <circle cx={node.x + 16} cy={node.y + 20} r={5} fill={color} />
                  <text x={node.x + 28} y={node.y + 17} fill="var(--text-primary)" fontSize={10} fontWeight={600}>
                    {node.label.length > 18 ? node.label.slice(0, 18) + '...' : node.label}
                  </text>
                  <text x={node.x + 28} y={node.y + 30} fill="#64748b" fontSize={9}>
                    {node.type === 'dataset' ? (node as any).version : node.type === 'pipeline' ? (node as any).status : node.type === 'model' ? (node as any).stage : (node as any).traffic || ''}
                  </text>
                </g>
              );
            })}

            {/* Column Headers */}
            <text x={120} y={30} fill="#64748b" fontSize={11} fontWeight={600} textAnchor="middle">DATASETS</text>
            <text x={450} y={30} fill="#64748b" fontSize={11} fontWeight={600} textAnchor="middle">PIPELINES</text>
            <text x={750} y={30} fill="#64748b" fontSize={11} fontWeight={600} textAnchor="middle">MODELS</text>
            <text x={990} y={30} fill="#64748b" fontSize={11} fontWeight={600} textAnchor="middle">ENDPOINTS</text>
          </svg>

          {/* Detail panel for selected node */}
          {selectedNode && (
            <div style={{
              position: 'absolute', top: 0, right: 0, width: 280, height: '100%',
              background: 'rgba(10,14,26,0.95)', borderLeft: '1px solid rgba(255,255,255,0.08)',
              padding: 16, display: 'flex', flexDirection: 'column', gap: 10,
              backdropFilter: 'blur(12px)', overflowY: 'auto',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <NodeIcon type={selectedNode.type} />
                <span style={{ fontWeight: 600, fontSize: 14 }}>{selectedNode.label}</span>
              </div>
              <div style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 600, color: getNodeColor(selectedNode.type), letterSpacing: '0.06em' }}>
                {selectedNode.type}
              </div>
              {selectedNode.type === 'dataset' && (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)' }}>
                    <span>Version</span><span style={{ fontFamily: 'var(--font-mono)' }}>{(selectedNode as any).version}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)' }}>
                    <span>Size</span><span>{(selectedNode as any).size}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)' }}>
                    <span>Records</span><span style={{ fontFamily: 'var(--font-mono)' }}>{(selectedNode as any).records}</span>
                  </div>
                </>
              )}
              <div style={{ fontSize: 12, marginTop: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Upstream ({upstreamIds.length})</div>
                {upstreamIds.map(id => {
                  const n = NODES.find(n => n.id === id);
                  return (
                    <div key={id} style={{ padding: '4px 0', fontSize: 12, color: 'var(--accent-secondary)', cursor: 'pointer' }}
                      onClick={() => setSelected(id)}>
                      ← {n?.label}
                    </div>
                  );
                })}
                {upstreamIds.length === 0 && <div style={{ color: 'var(--text-dimmed)', fontSize: 11 }}>None (source)</div>}
              </div>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Downstream ({downstreamIds.length})</div>
                {downstreamIds.map(id => {
                  const n = NODES.find(n => n.id === id);
                  return (
                    <div key={id} style={{ padding: '4px 0', fontSize: 12, color: 'var(--accent-primary)', cursor: 'pointer' }}
                      onClick={() => setSelected(id)}>
                      → {n?.label}
                    </div>
                  );
                })}
                {downstreamIds.length === 0 && <div style={{ color: 'var(--text-dimmed)', fontSize: 11 }}>None (leaf)</div>}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Datasets Table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Registered Datasets</span>
          <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{datasets.length} datasets</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Version</th>
              <th>Format</th>
              <th>Size</th>
              <th>Records</th>
              <th>Source</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((ds) => (
              <tr key={ds.name} style={{ cursor: 'pointer' }}
                onClick={() => setSelected(NODES.find(n => n.label === ds.name)?.id || null)}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Database style={{ width: 14, height: 14, color: 'var(--accent-secondary)' }} />
                    <span style={{ fontWeight: 500 }}>{ds.name}</span>
                  </div>
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{ds.version}</td>
                <td>
                  <span style={{
                    fontSize: 11, fontWeight: 600, color: 'var(--accent-primary)',
                    background: 'rgba(124,58,237,0.12)', padding: '2px 8px', borderRadius: 100,
                  }}>{ds.format}</span>
                </td>
                <td>{ds.size}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>{ds.records}</td>
                <td style={{ color: 'var(--text-muted)' }}>{ds.source}</td>
                <td style={{ color: 'var(--text-muted)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Clock style={{ width: 12, height: 12 }} />
                    {ds.updated}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
