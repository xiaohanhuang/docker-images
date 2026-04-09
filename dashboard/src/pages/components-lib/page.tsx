
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Layers, Box, Settings2, Loader2, X, Tag, Package, ArrowLeft, Search, GitBranch, Cpu, Filter } from 'lucide-react';
import { api } from '@/lib/api';

interface Component {
  name: string;
  version: string;
  desc: string;
  type: string;
  category?: string;
  tags?: string[];
}

interface ComponentDetail {
  name: string;
  version: string;
  desc: string;
  category: string;
  tags: string[];
  image: string;
  image_tag: string;
  inputs: { name: string; type: string }[];
  outputs: { name: string; type: string }[];
  task_type: string;
}

const CATEGORY_COLORS: Record<string, string> = {
  data: '#3b82f6',
  training: '#f59e0b',
  evaluation: '#10b981',
  genai: '#8b5cf6',
  model: '#ec4899',
  serving: '#06b6d4',
  ops: '#6b7280',
};

function CategoryBadge({ category }: { category: string }) {
  const color = CATEGORY_COLORS[category] || '#6b7280';
  return (
    <span style={{ fontSize: 11, color, backgroundColor: `${color}20`, padding: '2px 8px', borderRadius: 4, textTransform: 'uppercase', fontWeight: 600 }}>
      {category}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const isWorkflow = type === 'workflow';
  const color = isWorkflow ? '#a78bfa' : '#34d399';
  const Icon = isWorkflow ? GitBranch : Cpu;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      fontSize: 11, color, backgroundColor: `${color}20`,
      padding: '2px 8px', borderRadius: 4, textTransform: 'uppercase', fontWeight: 600,
    }}>
      <Icon size={11} />
      {type}
    </span>
  );
}

function FilterChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        fontSize: 12, fontWeight: 500,
        padding: '4px 12px', borderRadius: 16,
        border: active ? '1px solid var(--accent-primary)' : '1px solid rgba(255,255,255,0.12)',
        backgroundColor: active ? 'rgba(99,102,241,0.15)' : 'transparent',
        color: active ? 'var(--accent-primary)' : 'var(--text-dimmed)',
        cursor: 'pointer', transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  );
}

function ComponentDetailPanel({ name, onClose }: { name: string; onClose: () => void }) {
  const { data, isLoading, error } = useQuery<ComponentDetail>({
    queryKey: ['component', name],
    queryFn: () => api.components.get(name),
  });

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', justifyContent: 'flex-end' }}>
      <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.5)' }} onClick={onClose} />
      <div style={{ position: 'relative', width: 480, maxWidth: '100%', backgroundColor: 'var(--bg-primary)', borderLeft: '1px solid var(--border-primary)', overflowY: 'auto', padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dimmed)', padding: 4 }}>
            <ArrowLeft size={20} />
          </button>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Component Details</h2>
        </div>

        {isLoading ? (
          <div style={{ padding: 40, textAlign: 'center' }}>
            <Loader2 className="animate-spin" style={{ margin: '0 auto', color: 'var(--text-dimmed)' }} />
          </div>
        ) : error ? (
          <div className="error-state">Failed to load component details.</div>
        ) : data ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div>
              <div style={{ fontSize: 24, fontWeight: 700 }}>{data.name}</div>
              <div style={{ marginTop: 8, fontSize: 14, color: 'var(--text-dimmed)' }}>{data.desc}</div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div className="stat-card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 4 }}>Version</div>
                <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace' }}>{data.version}</div>
              </div>
              <div className="stat-card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 4 }}>Category</div>
                {data.category ? <CategoryBadge category={data.category} /> : <span style={{ color: 'var(--text-dimmed)' }}>—</span>}
              </div>
              <div className="stat-card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 4 }}>Type</div>
                <TypeBadge type={data.task_type === 'workflow' ? 'workflow' : 'task'} />
              </div>
            </div>

            {data.task_type && (
              <div className="stat-card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 4 }}>Task Type</div>
                <div style={{ fontFamily: 'monospace', fontSize: 14 }}>{data.task_type}</div>
              </div>
            )}

            {data.image && (
              <div className="stat-card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 8 }}>
                  <Package size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
                  Container Image
                </div>
                <div style={{ fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>{data.image}</div>
              </div>
            )}

            {data.inputs && data.inputs.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 8, fontWeight: 600 }}>
                  Inputs
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-dimmed)', fontWeight: 600 }}>Parameter</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-dimmed)', fontWeight: 600 }}>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.inputs.map(inp => (
                      <tr key={inp.name} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace', color: '#67e8f9' }}>{inp.name}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace', color: '#fbbf24' }}>{inp.type}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {data.outputs && data.outputs.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 8, fontWeight: 600 }}>
                  Outputs
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-dimmed)', fontWeight: 600 }}>Parameter</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-dimmed)', fontWeight: 600 }}>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.outputs.map(out => (
                      <tr key={out.name} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace', color: '#67e8f9' }}>{out.name}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace', color: '#fbbf24' }}>{out.type}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {data.tags && data.tags.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginBottom: 8 }}>
                  <Tag size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
                  Tags
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {data.tags.map(tag => (
                    <span key={tag} style={{ fontSize: 12, padding: '3px 10px', borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.06)', color: 'var(--text-dimmed)' }}>
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function ComponentsPage() {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['components'],
    queryFn: () => api.components.list(),
  });

  const components: Component[] = data?.components || [];

  const categories = useMemo(() => {
    const cats = new Set<string>();
    components.forEach(c => { if (c.category) cats.add(c.category); });
    return Array.from(cats).sort();
  }, [components]);

  const types = useMemo(() => {
    const ts = new Set<string>();
    components.forEach(c => ts.add(c.type));
    return Array.from(ts).sort();
  }, [components]);

  const filtered = useMemo(() => {
    return components.filter(c => {
      if (typeFilter && c.type !== typeFilter) return false;
      if (categoryFilter && c.category !== categoryFilter) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        return c.name.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q);
      }
      return true;
    });
  }, [components, searchQuery, typeFilter, categoryFilter]);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Component Library</h1>
        <p>Reusable ML pipeline components with typed interfaces</p>
      </div>

      {/* Filter Bar */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 16 }}>
        {/* Search */}
        <div style={{ position: 'relative', maxWidth: 400 }}>
          <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dimmed)' }} />
          <input
            type="text"
            placeholder="Search components..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            style={{
              width: '100%', padding: '8px 12px 8px 36px',
              backgroundColor: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 8, color: 'var(--text-primary)', fontSize: 14, outline: 'none',
            }}
          />
        </div>

        {/* Filter chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <Filter size={14} style={{ color: 'var(--text-dimmed)', marginRight: 4 }} />

          {/* Type filters */}
          <span style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginRight: 2 }}>Type:</span>
          <FilterChip label="All" active={typeFilter === null} onClick={() => setTypeFilter(null)} />
          {types.map(t => (
            <FilterChip key={t} label={t} active={typeFilter === t} onClick={() => setTypeFilter(typeFilter === t ? null : t)} />
          ))}

          <span style={{ width: 1, height: 20, backgroundColor: 'rgba(255,255,255,0.1)', margin: '0 8px' }} />

          {/* Category filters */}
          <span style={{ fontSize: 11, color: 'var(--text-dimmed)', textTransform: 'uppercase', marginRight: 2 }}>Category:</span>
          <FilterChip label="All" active={categoryFilter === null} onClick={() => setCategoryFilter(null)} />
          {categories.map(cat => (
            <FilterChip key={cat} label={cat} active={categoryFilter === cat} onClick={() => setCategoryFilter(categoryFilter === cat ? null : cat)} />
          ))}
        </div>

        {/* Results count */}
        <div style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>
          {filtered.length} of {components.length} components
        </div>
      </div>

      {isLoading ? (
        <div style={{ padding: '40px', textAlign: 'center' }}>
          <Loader2 className="animate-spin" style={{ margin: '0 auto', color: 'var(--text-dimmed)' }} />
        </div>
      ) : error ? (
        <div className="error-state">Failed to load components.</div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', marginTop: 16 }}>
          {filtered.map(c => {
            const isWorkflow = c.type === 'workflow';
            const borderAccent = isWorkflow ? 'rgba(167,139,250,0.3)' : 'rgba(52,211,153,0.3)';
            return (
              <div
                key={c.name}
                className="stat-card"
                onClick={() => setSelectedComponent(c.name)}
                style={{
                  display: 'flex', flexDirection: 'column', gap: 12,
                  cursor: 'pointer', transition: 'border-color 0.15s',
                  borderLeft: `3px solid ${borderAccent}`,
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent-primary)')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = '')}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  {isWorkflow
                    ? <GitBranch style={{ color: '#a78bfa', width: 18, flexShrink: 0 }} />
                    : <Cpu style={{ color: '#34d399', width: 18, flexShrink: 0 }} />
                  }
                  <div style={{ fontWeight: 600, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-dimmed)', backgroundColor: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: 4, fontFamily: 'monospace', flexShrink: 0 }}>
                    v{c.version}
                  </div>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-dimmed)', flex: 1 }}>{c.desc}</div>
                <div style={{ marginTop: 'auto', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <TypeBadge type={c.type} />
                  {c.category && <CategoryBadge category={c.category} />}
                </div>
              </div>
            );
          })}
          {filtered.length === 0 && (
             <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: 'var(--text-dimmed)' }}>
               {components.length === 0 ? 'No components found in the connected registry.' : 'No components match the current filters.'}
             </div>
          )}
        </div>
      )}

      {selectedComponent && (
        <ComponentDetailPanel name={selectedComponent} onClose={() => setSelectedComponent(null)} />
      )}
    </div>
  );
}
