
import { useState, useEffect, useRef, useCallback, KeyboardEvent } from 'react';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';

/* ──────────── Hardware ──── */
const GPU_OPTIONS = [
  { type: 'CPU',  ratePerGpu: 0.192 / 4, validGpus: [] as number[] },
  { type: 'T4',   ratePerGpu: 0.526,     validGpus: [0.25, 0.5, 1, 2, 4, 8] },
  { type: 'A10G', ratePerGpu: 1.006,     validGpus: [0.25, 0.5, 1, 2, 4, 8] },
  { type: 'A100', ratePerGpu: 4.10,      validGpus: [0.25, 0.5, 1, 2, 4, 8] },
  { type: 'H100', ratePerGpu: 6.88,      validGpus: [0.25, 0.5, 1, 2, 4, 8] },
  { type: 'H100', ratePerGpu: 6.88,      validGpus: [8] },
];

/* ──────────── IDE Tool definitions ──── */
type IDETool = 'vscode' | 'notebook' | 'marimo';

const IDE_TOOLS: { id: IDETool; label: string; icon: string; color: string }[] = [
  { id: 'vscode', label: 'VS Code', icon: '💻', color: '#007ACC' },
  { id: 'notebook', label: 'Notebook', icon: '📓', color: '#F37626' },
  { id: 'marimo', label: 'Marimo', icon: '🌊', color: '#2CB1BC' },
];

/* ──────────── Cell types ──── */
type CellType = 'code' | 'markdown';

interface CellOutput {
  stdout: string;
  stderr: string;
  images: string[];
}

interface Cell {
  id: string;
  type: CellType;
  source: string;
  output: CellOutput | null;
  executing: boolean;
  executionCount: number | null;
}

let cellIdCounter = 0;
function newCell(type: CellType = 'code', source = ''): Cell {
  return {
    id: `cell-${++cellIdCounter}-${Date.now()}`,
    type,
    source,
    output: null,
    executing: false,
    executionCount: null,
  };
}

/* ================================================================== */
/*  NOTEBOOK CELL COMPONENT                                            */
/* ================================================================== */

function NotebookCell({
  cell, isActive, onFocus, onUpdate, onExecute, onDelete, onInsertBelow,
  onMoveUp, onMoveDown, isFirst, isLast,
}: {
  cell: Cell; isActive: boolean;
  onFocus: () => void; onUpdate: (source: string) => void;
  onExecute: () => void; onDelete: () => void; onInsertBelow: () => void;
  onMoveUp: () => void; onMoveDown: () => void;
  isFirst: boolean; isLast: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [localSource, setLocalSource] = useState(cell.source);

  useEffect(() => { setLocalSource(cell.source); }, [cell.source]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.max(ta.scrollHeight, 36) + 'px';
    }
  }, [localSource]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.shiftKey || e.ctrlKey)) {
      e.preventDefault();
      onUpdate(localSource);
      onExecute();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      const ta = e.target as HTMLTextAreaElement;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const val = localSource;
      setLocalSource(val.substring(0, start) + '    ' + val.substring(end));
      setTimeout(() => { ta.selectionStart = ta.selectionEnd = start + 4; }, 0);
    }
  };

  const accentBorder = isActive ? 'rgba(124,58,237,0.5)' : 'rgba(255,255,255,0.06)';
  const executionLabel = cell.executionCount !== null ? `[${cell.executionCount}]` : '[ ]';

  return (
    <div
      onClick={onFocus}
      style={{
        marginBottom: 2,
        borderLeft: `3px solid ${accentBorder}`,
        transition: 'border-color 0.2s',
        background: isActive ? 'rgba(124,58,237,0.02)' : 'transparent',
      }}
    >
      {/* Cell header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '4px 12px 4px 8px',
        opacity: isActive ? 1 : 0.6,
        transition: 'opacity 0.2s',
      }}>
        {/* Execution count / run button */}
        <button
          onClick={(e) => { e.stopPropagation(); onUpdate(localSource); onExecute(); }}
          title="Run cell (Shift+Enter)"
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 48, minWidth: 48,
            background: 'none', border: 'none', cursor: 'pointer',
            color: cell.executing ? '#a78bfa' : 'var(--text-dimmed)',
            fontFamily: '"JetBrains Mono", monospace', fontSize: 11, fontWeight: 600,
          }}
        >
          {cell.executing ? (
            <div style={{
              width: 14, height: 14, borderRadius: '50%',
              border: '2px solid rgba(167,139,250,0.3)',
              borderTopColor: '#a78bfa',
              animation: 'spin 0.8s linear infinite',
            }} />
          ) : cell.type === 'code' ? executionLabel : 'Md'}
        </button>

        {/* Cell type badge */}
        <select
          value={cell.type}
          onChange={(e) => {
            const newType = e.target.value as CellType;
            onUpdate(localSource);
            // Type change handled upstream
          }}
          style={{
            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 4, color: 'var(--text-dimmed)', fontSize: 10, padding: '2px 4px',
            cursor: 'pointer', fontFamily: '"JetBrains Mono", monospace',
          }}
        >
          <option value="code">Code</option>
          <option value="markdown">Markdown</option>
        </select>

        <div style={{ flex: 1 }} />

        {/* Cell actions */}
        {isActive && (
          <div style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <CellButton icon="▲" onClick={onMoveUp} disabled={isFirst} title="Move up" />
            <CellButton icon="▼" onClick={onMoveDown} disabled={isLast} title="Move down" />
            <CellButton icon="＋" onClick={onInsertBelow} title="Insert cell below" />
            <CellButton icon="✕" onClick={onDelete} title="Delete cell" />
          </div>
        )}
      </div>

      {/* Cell editor */}
      <div style={{ padding: '0 12px 0 8px', display: 'flex' }}>
        <div style={{ width: 48, minWidth: 48 }} /> {/* gutter spacer */}
        <div style={{
          flex: 1,
          background: 'rgba(15,20,35,0.5)',
          borderRadius: 6,
          border: `1px solid ${isActive ? 'rgba(124,58,237,0.2)' : 'rgba(255,255,255,0.04)'}`,
          overflow: 'hidden',
        }}>
          <textarea
            ref={textareaRef}
            value={localSource}
            onChange={(e) => setLocalSource(e.target.value)}
            onBlur={() => onUpdate(localSource)}
            onKeyDown={handleKeyDown}
            onFocus={onFocus}
            placeholder={cell.type === 'code' ? '# Enter Python code...' : '# Markdown text...'}
            spellCheck={false}
            style={{
              width: '100%', resize: 'none', overflow: 'hidden',
              background: 'transparent', border: 'none', outline: 'none',
              color: '#e2e8f0', fontFamily: '"JetBrains Mono", monospace',
              fontSize: 13, lineHeight: 1.6, padding: '8px 12px',
              minHeight: 36,
            }}
          />
        </div>
      </div>

      {/* Cell output */}
      {cell.output && (cell.output.stdout || cell.output.stderr || cell.output.images.length > 0) && (
        <div style={{ padding: '4px 12px 8px 8px', display: 'flex' }}>
          <div style={{ width: 48, minWidth: 48 }} />
          <div style={{
            flex: 1, borderRadius: 6, overflow: 'hidden',
            border: `1px solid ${cell.output.stderr ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.04)'}`,
            background: cell.output.stderr ? 'rgba(239,68,68,0.03)' : 'rgba(15,20,35,0.3)',
          }}>
            {/* Text output */}
            {cell.output.stdout && (
              <pre style={{
                margin: 0, padding: '8px 12px',
                fontFamily: '"JetBrains Mono", monospace', fontSize: 12,
                lineHeight: 1.5, color: '#94a3b8', whiteSpace: 'pre-wrap',
                wordBreak: 'break-word', maxHeight: 400, overflow: 'auto',
              }}>{cell.output.stdout}</pre>
            )}
            {/* Error output */}
            {cell.output.stderr && (
              <pre style={{
                margin: 0, padding: '8px 12px',
                fontFamily: '"JetBrains Mono", monospace', fontSize: 12,
                lineHeight: 1.5, color: '#f87171', whiteSpace: 'pre-wrap',
                wordBreak: 'break-word', maxHeight: 400, overflow: 'auto',
              }}>{cell.output.stderr}</pre>
            )}
            {/* Image outputs (matplotlib plots) */}
            {cell.output.images.map((img, i) => (
              <div key={i} style={{ padding: '8px 12px', textAlign: 'center' }}>
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Plot ${i + 1}`}
                  style={{ maxWidth: '100%', borderRadius: 4, background: '#0f1423' }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function CellButton({ icon, onClick, disabled, title }: {
  icon: string; onClick: () => void; disabled?: boolean; title: string;
}) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      disabled={disabled}
      title={title}
      style={{
        background: 'none', border: 'none', cursor: disabled ? 'default' : 'pointer',
        color: disabled ? 'rgba(255,255,255,0.15)' : 'var(--text-dimmed)',
        fontSize: 12, padding: '2px 5px', borderRadius: 3, lineHeight: 1,
        opacity: disabled ? 0.4 : 0.7,
      }}
    >{icon}</button>
  );
}

/* ================================================================== */
/*  NOTEBOOK TOOLBAR                                                    */
/* ================================================================== */

function NotebookToolbar({
  onAddCell, onRunAll, onClearOutputs, onSave, cellCount,
}: {
  onAddCell: (type: CellType) => void; onRunAll: () => void;
  onClearOutputs: () => void; onSave: () => void; cellCount: number;
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '6px 16px',
      borderBottom: '1px solid rgba(255,255,255,0.06)',
      background: 'rgba(8,12,22,0.6)',
    }}>
      <ToolbarButton onClick={() => onAddCell('code')} icon="＋" label="Code" />
      <ToolbarButton onClick={() => onAddCell('markdown')} icon="¶" label="Markdown" />
      <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />
      <ToolbarButton onClick={onRunAll} icon="▶" label="Run All" accent />
      <ToolbarButton onClick={onClearOutputs} icon="◯" label="Clear" />
      <div style={{ flex: 1 }} />
      <span style={{ fontSize: 11, color: 'var(--text-dimmed)', fontFamily: '"JetBrains Mono", monospace' }}>
        {cellCount} cell{cellCount !== 1 ? 's' : ''}
      </span>
      <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.08)', margin: '0 4px' }} />
      <ToolbarButton onClick={onSave} icon="💾" label="Save" />
    </div>
  );
}

function ToolbarButton({ onClick, icon, label, accent }: {
  onClick: () => void; icon: string; label: string; accent?: boolean;
}) {
  return (
    <button onClick={onClick} style={{
      display: 'flex', alignItems: 'center', gap: 5,
      padding: '4px 10px', borderRadius: 5,
      background: accent ? 'rgba(124,58,237,0.12)' : 'rgba(255,255,255,0.04)',
      border: `1px solid ${accent ? 'rgba(124,58,237,0.2)' : 'rgba(255,255,255,0.06)'}`,
      color: accent ? '#a78bfa' : 'var(--text-dimmed)',
      fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500,
      transition: 'all 0.15s',
    }}>
      <span>{icon}</span>
      <span>{label}</span>
    </button>
  );
}

/* ================================================================== */
/*  NATIVE NOTEBOOK                                                     */
/* ================================================================== */

function NativeNotebook({ deskId, isVisible }: { deskId: string; isVisible: boolean }) {
  const [cells, setCells] = useState<Cell[]>([
    newCell('code', '# Welcome to your notebook\n# Press Shift+Enter to run a cell\nprint("Hello from AI Desk!")'),
    newCell('code', ''),
  ]);
  const [activeCell, setActiveCell] = useState<string>(cells[0].id);
  const executionCountRef = useRef(0);

  const updateCellSource = useCallback((cellId: string, source: string) => {
    setCells(prev => prev.map(c => c.id === cellId ? { ...c, source } : c));
  }, []);

  const executeCell = useCallback(async (cellId: string) => {
    const cell = cells.find(c => c.id === cellId);
    if (!cell || cell.type !== 'code' || !cell.source.trim()) return;

    setCells(prev => prev.map(c => c.id === cellId
      ? { ...c, executing: true, output: null }
      : c
    ));

    try {
      const res = await fetch(`/api/v1/desks/${deskId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: cell.source }),
      });
      const data = await res.json();
      const execCount = ++executionCountRef.current;
      setCells(prev => prev.map(c => c.id === cellId
        ? {
            ...c,
            executing: false,
            executionCount: execCount,
            output: {
              stdout: data.stdout || '',
              stderr: data.stderr || '',
              images: data.images || [],
            },
          }
        : c
      ));
    } catch (err: any) {
      setCells(prev => prev.map(c => c.id === cellId
        ? {
            ...c,
            executing: false,
            output: { stdout: '', stderr: `Request failed: ${err.message}`, images: [] },
          }
        : c
      ));
    }
  }, [cells, deskId]);

  const addCell = useCallback((type: CellType, afterId?: string) => {
    const cell = newCell(type);
    setCells(prev => {
      if (!afterId) return [...prev, cell];
      const idx = prev.findIndex(c => c.id === afterId);
      const arr = [...prev];
      arr.splice(idx + 1, 0, cell);
      return arr;
    });
    setActiveCell(cell.id);
  }, []);

  const deleteCell = useCallback((cellId: string) => {
    setCells(prev => {
      if (prev.length <= 1) return prev; // Keep at least one cell
      const idx = prev.findIndex(c => c.id === cellId);
      const next = prev.filter(c => c.id !== cellId);
      if (cellId === activeCell) {
        setActiveCell(next[Math.min(idx, next.length - 1)].id);
      }
      return next;
    });
  }, [activeCell]);

  const moveCell = useCallback((cellId: string, direction: 'up' | 'down') => {
    setCells(prev => {
      const idx = prev.findIndex(c => c.id === cellId);
      if ((direction === 'up' && idx === 0) || (direction === 'down' && idx === prev.length - 1)) return prev;
      const arr = [...prev];
      const swap = direction === 'up' ? idx - 1 : idx + 1;
      [arr[idx], arr[swap]] = [arr[swap], arr[idx]];
      return arr;
    });
  }, []);

  const runAll = useCallback(async () => {
    for (const cell of cells) {
      if (cell.type === 'code' && cell.source.trim()) {
        await executeCell(cell.id);
      }
    }
  }, [cells, executeCell]);

  const clearOutputs = useCallback(() => {
    setCells(prev => prev.map(c => ({ ...c, output: null, executionCount: null })));
    executionCountRef.current = 0;
  }, []);

  if (!isVisible) return null;

  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      background: 'var(--bg-primary, #080c16)',
    }}>
      <NotebookToolbar
        onAddCell={(t) => addCell(t)}
        onRunAll={runAll}
        onClearOutputs={clearOutputs}
        onSave={() => {/* TODO: save .ipynb */}}
        cellCount={cells.length}
      />
      <div style={{
        flex: 1, overflow: 'auto', padding: '12px 0',
      }}>
        <div style={{ maxWidth: 960, margin: '0 auto', padding: '0 16px' }}>
          {cells.map((cell, i) => (
            <NotebookCell
              key={cell.id}
              cell={cell}
              isActive={cell.id === activeCell}
              onFocus={() => setActiveCell(cell.id)}
              onUpdate={(src) => updateCellSource(cell.id, src)}
              onExecute={() => executeCell(cell.id)}
              onDelete={() => deleteCell(cell.id)}
              onInsertBelow={() => addCell('code', cell.id)}
              onMoveUp={() => moveCell(cell.id, 'up')}
              onMoveDown={() => moveCell(cell.id, 'down')}
              isFirst={i === 0}
              isLast={i === cells.length - 1}
            />
          ))}
          {/* Add cell button at bottom */}
          <div style={{ display: 'flex', justifyContent: 'center', padding: '16px 0' }}>
            <button
              onClick={() => addCell('code')}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 20px', borderRadius: 20,
                background: 'rgba(255,255,255,0.03)',
                border: '1px dashed rgba(255,255,255,0.1)',
                color: 'var(--text-dimmed)', fontSize: 12,
                cursor: 'pointer', fontFamily: 'inherit',
                transition: 'all 0.2s',
              }}
            >
              ＋ Add Cell
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  SHARED UI COMPONENTS                                               */
/* ================================================================== */

function GpuDropdown({ gpuType, onChange, open, onToggle }: {
  gpuType: string; onChange: (t: string) => void; open: boolean; onToggle: () => void;
}) {
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={onToggle} style={{
        display: 'flex', alignItems: 'center', gap: 6,
        background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 6, padding: '6px 12px', cursor: 'pointer',
        color: 'var(--text-primary)', fontSize: 14,
        fontFamily: '"JetBrains Mono", monospace', fontWeight: 600,
      }}>
        {gpuType}
        <svg width="10" height="6" viewBox="0 0 10 6" fill="none" style={{ transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : '' }}>
          <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      {open && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, marginTop: 4, zIndex: 100,
          background: 'rgba(15,20,35,0.98)', border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 8, boxShadow: '0 12px 32px rgba(0,0,0,0.5)', minWidth: 140, overflow: 'hidden',
        }}>
          {GPU_OPTIONS.map(opt => (
            <button key={opt.type} onClick={(e) => { e.stopPropagation(); onToggle(); onChange(opt.type); }} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              width: '100%', padding: '8px 14px',
              background: opt.type === gpuType ? 'rgba(124,58,237,0.1)' : 'transparent',
              border: 'none', borderBottom: '1px solid rgba(255,255,255,0.04)',
              color: opt.type === gpuType ? 'var(--accent-primary)' : 'var(--text-primary)',
              fontSize: 13, fontFamily: '"JetBrains Mono", monospace',
              cursor: 'pointer', fontWeight: opt.type === gpuType ? 600 : 400,
            }}>
              <span>{opt.type}</span>
              <span style={{ color: 'var(--cost-green)', fontSize: 11 }}>${opt.ratePerGpu.toFixed(2)}/hr</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function GpuCountSelector({ count, onChange, validCounts }: { count: number; onChange: (n: number) => void; validCounts: number[] }) {
  if (validCounts.length <= 1) return null; // hide entirely if only 1 option
  const nearestSlot = validCounts.reduce((best, n) => (n <= count ? n : best), validCounts[0]);
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <span style={{ fontSize: 11, marginRight: 8, color: 'var(--text-dimmed)' }}>GPUs:</span>
      {validCounts.map((n, i) => {
        const isActive = n <= count;
        const isSelected = n === count || (count > 0 && n === nearestSlot && !validCounts.includes(count));
        return (
          <div key={n} style={{ display: 'flex', alignItems: 'center' }}>
            {i > 0 && (
              <div style={{
                width: 28, height: 3,
                background: n <= count ? 'var(--accent-primary)' : 'rgba(255,255,255,0.1)',
                transition: 'background 0.2s',
              }} />
            )}
            <button onClick={() => onChange(n)} style={{
              width: isSelected ? 28 : 22, height: isSelected ? 28 : 22,
              borderRadius: '50%',
              background: isSelected ? 'var(--accent-primary)' : isActive ? 'rgba(124,58,237,0.3)' : 'rgba(255,255,255,0.08)',
              border: '2px solid transparent',
              color: isSelected ? '#fff' : isActive ? 'var(--accent-primary)' : 'var(--text-dimmed)',
              fontSize: isSelected ? 13 : 11, fontWeight: 700, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.2s',
              boxShadow: isSelected ? '0 0 12px rgba(124,58,237,0.4)' : 'none',
            }}>
              {n}
            </button>
          </div>
        );
      })}
    </div>
  );
}

function CpuSlider({ count, onChange }: { count: number; onChange: (n: number) => void }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ fontSize: 11, color: 'var(--text-dimmed)' }}>CPUs:</span>
      <input type="range" min="1" max="256" value={count} onChange={e => onChange(Number(e.target.value))} style={{ width: 120 }} />
      <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{count}</span>
    </div>
  );
}



function MemSlider({ count, onChange }: { count: number; onChange: (n: number) => void }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ fontSize: 11, color: 'var(--text-dimmed)' }}>MEM:</span>
      <input type="range" min="0.5" max="256" step="0.5" value={count} onChange={e => onChange(Number(e.target.value))} style={{ width: 120 }} />
      <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{count < 1 ? '500MB' : `${count}GB`}</span>
    </div>
  );
}

function IDETabBar({ activeTool, onSelect }: {
  activeTool: IDETool; onSelect: (tool: IDETool) => void;
}) {
  return (
    <div style={{
      display: 'flex', gap: 2, padding: '0 4px',
      background: 'rgba(255,255,255,0.02)', borderRadius: 8,
    }}>
      {IDE_TOOLS.map(tool => {
        const isActive = activeTool === tool.id;
        return (
          <button
            key={tool.id}
            onClick={() => onSelect(tool.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 14px',
              background: isActive
                ? `linear-gradient(135deg, ${tool.color}22, ${tool.color}11)`
                : 'transparent',
              border: 'none',
              borderBottom: isActive ? `2px solid ${tool.color}` : '2px solid transparent',
              borderRadius: '6px 6px 0 0',
              color: isActive ? '#fff' : 'var(--text-dimmed)',
              fontSize: 13, fontWeight: isActive ? 600 : 400,
              cursor: 'pointer', transition: 'all 0.2s ease',
              fontFamily: 'inherit',
            }}
          >
            <span style={{ fontSize: 14 }}>{tool.icon}</span>
            <span>{tool.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/* ================================================================== */
/*  MAIN PAGE                                                          */
/* ================================================================== */

export default function DeskIDEPage() {
  const params = useParams();
  const navigate = useNavigate();
  const deskId = params.id as string;

  const [gpuType, setGpuType] = useState('CPU');
  const [gpuCount, setGpuCount] = useState(0);
  const [cpuCount, setCpuCount] = useState(1);
  const [memCount, setMemCount] = useState(16);
  const [gpuDropdownOpen, setGpuDropdownOpen] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);
  const [pendingChange, setPendingChange] = useState<{ type: string; gpuCount: number; cpuCount: number; memCount: number } | null>(null);

  const [searchParams] = useSearchParams();
  const initialTab = (searchParams.get('tab') || 'vscode') as IDETool;
  const [activeTool, setActiveTool] = useState<IDETool>(initialTab);

  // VS Code state
  const [vscodeReady, setVscodeReady] = useState(false);
  const [vscodeLoading, setVscodeLoading] = useState(true);
  const [vscodeError, setVscodeError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState('Starting VS Code Server...');

  const selectedOpt = GPU_OPTIONS.find(o => o.type === gpuType) || GPU_OPTIONS[0];
  const burnRate = selectedOpt.ratePerGpu * (gpuType === 'CPU' ? cpuCount : gpuCount);

  // Fetch actual desk info to populate GPU selector
  useEffect(() => {
    async function fetchDeskInfo() {
      try {
        const res = await fetch('/api/v1/desks');
        if (!res.ok) return;
        const data = await res.json();
        const desk = data.desks?.find((d: { id: string }) => d.id === deskId);
        if (!desk) return;
        // Parse gpu field: "A100 x2" or "CPU"
        const gpuField = desk.gpu || 'CPU';
        if (gpuField === 'CPU') {
          setGpuType('CPU');
          setGpuCount(0);
        } else {
          const parts = gpuField.split(' x');
          setGpuType(parts[0] || 'CPU');
          setGpuCount(parseFloat(parts[1]) || 1);
        }
        // CPU/Mem count from API
        if (desk.cpu_count) setCpuCount(desk.cpu_count);
        if (desk.memory) {
          if (desk.memory === '500Mi') setMemCount(0.5);
          else setMemCount(parseFloat(desk.memory.replace('Gi', '')) || 16);
        }
      } catch { /* ignore */ }
    }
    fetchDeskInfo();
  }, [deskId]);

  // Handle GPU type/count changes — show confirmation modal
  const handleGpuTypeChange = useCallback((t: string) => {
    if (t === gpuType) return;
    const opt = GPU_OPTIONS.find(o => o.type === t) || GPU_OPTIONS[0];
    const newGpuCount = t === 'CPU' ? 0 : opt.validGpus.length > 0 ? opt.validGpus[0] : 1;
    setPendingChange({ type: t, gpuCount: newGpuCount, cpuCount: cpuCount, memCount });
  }, [gpuType, cpuCount]);

  const handleGpuCountChange = useCallback((c: number) => {
    if (c === gpuCount) return;
    setPendingChange({ type: gpuType, gpuCount: c, cpuCount: cpuCount, memCount });
  }, [gpuType, gpuCount, cpuCount]);

  const handleCpuCountChange = useCallback((c: number) => {
    if (c === cpuCount) return;
    setPendingChange({ type: gpuType, gpuCount, cpuCount: c, memCount });
  }, [gpuType, gpuCount, cpuCount, memCount]);

  const handleMemCountChange = useCallback((c: number) => {
    if (c === memCount) return;
    setPendingChange({ type: gpuType, gpuCount, cpuCount, memCount: c });
  }, [gpuType, gpuCount, cpuCount, memCount]);

  // Execute the actual restart
  const executeRestart = useCallback(async () => {
    if (!pendingChange) return;
    const { type: newType, gpuCount: newGpuCount, cpuCount: newCpuCount, memCount: newMemCount } = pendingChange;
    setPendingChange(null);
    setIsRestarting(true);
    setStatusMessage('Restarting desk with new resources...');
    setVscodeReady(false);
    setVscodeLoading(true);
    try {
      await fetch(`/api/v1/desks/${deskId}`, { method: 'DELETE' });
      await new Promise(r => setTimeout(r, 5000));
      const name = deskId.replace('desk-', '');
      await fetch('/api/v1/desks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          gpu_type: newType,
          gpu_count: newType === 'CPU' ? 0 : Math.round(newGpuCount * 4),
          cpu_count: newType === 'CPU' ? newCpuCount : Math.max(4, Math.round(newGpuCount * 4)),
          memory: newMemCount === 0.5 ? '500Mi' : `${newMemCount}Gi`,
        }),
      });
      window.location.reload();
    } catch (err) {
      alert(`Failed to restart desk: ${err}`);
      setIsRestarting(false);
      setVscodeLoading(false);
    }
  }, [deskId, pendingChange]);

  // Hide global sidebar for immersive IDE
  useEffect(() => {
    const sidebar = document.querySelector('.app-sidebar') as HTMLElement;
    if (sidebar) sidebar.style.display = 'none';
    return () => { if (sidebar) sidebar.style.display = ''; };
  }, []);

  // Start code-server
  useEffect(() => {
    let cancelled = false;
    let retryCount = 0;
    const maxRetries = 60;

    async function launchVscode() {
      setVscodeLoading(true);
      setVscodeError(null);
      setStatusMessage('Starting VS Code Server...');

      while (retryCount < maxRetries && !cancelled) {
        try {
          const res = await fetch(`/api/v1/desks/${deskId}/start-vscode`, { method: 'POST' });
          if (res.ok) break;
          if (res.status === 409) {
            const body = await res.json().catch(() => ({ detail: 'Pod is starting up...' }));
            setStatusMessage(body.detail || 'Pod is starting up — waiting for container to be ready...');
          } else {
            setStatusMessage(`Waiting for API (HTTP ${res.status})...`);
          }
          retryCount++;
          await new Promise(r => setTimeout(r, 3000));
        } catch (err: any) {
          retryCount++;
          setStatusMessage(`Connecting to API (attempt ${retryCount})...`);
          if (retryCount >= 20) {
            setVscodeError(`Could not connect to API after ${retryCount} attempts. ${err?.message || ''}`);
            setVscodeLoading(false);
            return;
          }
          if (!cancelled) await new Promise(r => setTimeout(r, 3000));
        }
      }
      if (cancelled) return;
      if (retryCount >= maxRetries) {
        setVscodeError('Pod took too long to start.');
        setVscodeLoading(false);
        return;
      }
      setStatusMessage('Connecting to IDE...');

      const proxyUrl = `/desk-proxy/${deskId}/`;
      for (let i = 0; i < 15 && !cancelled; i++) {
        try {
          await fetch(proxyUrl, { mode: 'no-cors', signal: AbortSignal.timeout(3000) });
          setVscodeReady(true);
          setVscodeLoading(false);
          return;
        } catch {
          await new Promise(r => setTimeout(r, 2000));
        }
      }
      if (!cancelled) {
        setVscodeError('K8s proxy connection timed out.');
        setVscodeLoading(false);
      }
    }

    launchVscode();
    return () => { cancelled = true; };
  }, [deskId]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!gpuDropdownOpen) return;
    const close = () => setGpuDropdownOpen(false);
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [gpuDropdownOpen]);

  const isReady = vscodeReady || activeTool === 'notebook';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-primary)' }}>
      {/* ═══════ HEADER BAR ═══════ */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 16px', height: 48, flexShrink: 0,
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(8,12,22,0.95)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={() => navigate('/desks')}
            style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none', border: 'none', color: 'var(--text-dimmed)', cursor: 'pointer', fontSize: 13 }}
            title="Back to Desks"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
          </button>
          <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.1)' }} />
          <div style={{
            width: 10, height: 10, borderRadius: '50%',
            background: '#10b981',
            boxShadow: '0 0 8px rgba(16,185,129,0.5)',
          }} />
          <span style={{ fontWeight: 600, fontSize: 16, color: 'var(--text-primary)' }}>{deskId}</span>
          <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.1)', marginLeft: 6 }} />
          <IDETabBar activeTool={activeTool} onSelect={setActiveTool} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }} onClick={e => e.stopPropagation()}>
          <GpuDropdown
            gpuType={gpuType} onChange={handleGpuTypeChange}
            open={gpuDropdownOpen} onToggle={() => setGpuDropdownOpen(o => !o)}
          />
          {gpuType === 'CPU' ? (
            <div style={{ display: 'flex', gap: 16 }}>
              <CpuSlider count={cpuCount} onChange={handleCpuCountChange} />
              <MemSlider count={memCount} onChange={handleMemCountChange} />
            </div>
          ) : (
            <GpuCountSelector count={gpuCount} onChange={handleGpuCountChange} validCounts={selectedOpt.validGpus} />
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            fontSize: 14, fontWeight: 600, color: 'var(--cost-green)',
            fontFamily: '"JetBrains Mono", monospace',
            background: 'rgba(52,211,153,0.08)',
            padding: '5px 14px', borderRadius: 100,
            border: '1px solid rgba(52,211,153,0.15)',
          }}>
            {gpuType} ${burnRate.toFixed(0)}/hr
          </div>
        </div>
      </div>

      {/* ═══════ MAIN WORKSPACE ═══════ */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {/* VS Code iframe — always alive in background */}
        {vscodeReady && (
          <iframe
            src={`/desk-proxy/${deskId}/`}
            style={{
              width: '100%', height: '100%', border: 'none',
              position: 'absolute', top: 0, left: 0,
              visibility: activeTool === 'vscode' ? 'visible' : 'hidden',
              zIndex: activeTool === 'vscode' ? 1 : 0,
            }}
            title="VS Code Workspace"
            sandbox="allow-scripts allow-same-origin allow-forms allow-modals allow-popups allow-downloads"
          />
        )}

        {/* Native Notebook */}
        <div style={{
          position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
          visibility: activeTool === 'notebook' ? 'visible' : 'hidden',
          zIndex: activeTool === 'notebook' ? 1 : 0,
        }}>
          <NativeNotebook deskId={deskId} isVisible={activeTool === 'notebook'} />
        </div>

        {/* Marimo iframe */}
        {vscodeReady && (
          <iframe
            src={`/desk-marimo/${deskId}/`}
            style={{
              width: '100%', height: '100%', border: 'none',
              position: 'absolute', top: 0, left: 0,
              visibility: activeTool === 'marimo' ? 'visible' : 'hidden',
              zIndex: activeTool === 'marimo' ? 1 : 0,
            }}
            title="Marimo Workspace"
          />
        )}

        {/* Loading / Error states for VS Code */}
        {activeTool === 'vscode' && !vscodeReady && (
          vscodeLoading ? (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
              <div style={{
                width: 48, height: 48, borderRadius: '50%',
                border: '3px solid rgba(124,58,237,0.2)',
                borderTopColor: 'var(--accent-primary)',
                animation: 'spin 1s linear infinite',
              }} />
              <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>{statusMessage}</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 340, textAlign: 'center', lineHeight: 1.6 }}>
                Launching code-server inside pod and connecting via K8s proxy
              </div>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          ) : vscodeError ? (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--error)" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
              <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>Failed to connect to IDE</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 340, textAlign: 'center', lineHeight: 1.6 }}>{vscodeError}</div>
              <button
                onClick={() => window.location.reload()}
                style={{
                  padding: '8px 20px', borderRadius: 6, fontSize: 13, fontWeight: 500,
                  background: 'var(--accent-primary)', border: 'none', color: '#fff', cursor: 'pointer',
                }}
              >Retry</button>
            </div>
          ) : null
        )}
      </div>

      {/* ═══════ RESOURCE CHANGE MODAL ═══════ */}
      {pendingChange && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }} onClick={() => setPendingChange(null)}>
          <div onClick={e => e.stopPropagation()} style={{
            background: 'rgba(20,24,40,0.98)', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 16, padding: '28px 32px', maxWidth: 420, width: '90%',
            boxShadow: '0 24px 64px rgba(0,0,0,0.6)',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>
              Change Resources?
            </div>
            <div style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 6 }}>
              Switch to <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
                                {pendingChange.type === 'CPU' ? `${pendingChange.cpuCount} CPUs (${pendingChange.memCount < 1 ? '500MB' : pendingChange.memCount + 'GB'})` : `${pendingChange.type} ×${pendingChange.gpuCount}`}
              </span>
            </div>
            <div style={{
              fontSize: 13, color: '#f59e0b', lineHeight: 1.6, marginBottom: 24,
              padding: '8px 12px', borderRadius: 8,
              background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
            }}>
              ⚠️ This will <strong>restart</strong> your desk. All unsaved work will be lost.
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button onClick={() => setPendingChange(null)} style={{
                padding: '8px 20px', borderRadius: 8, fontSize: 14, fontWeight: 500,
                background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
                color: 'var(--text-secondary)', cursor: 'pointer',
              }}>Cancel</button>
              <button onClick={executeRestart} style={{
                padding: '8px 20px', borderRadius: 8, fontSize: 14, fontWeight: 600,
                background: 'var(--accent-primary)', border: 'none',
                color: '#fff', cursor: 'pointer',
                boxShadow: '0 4px 12px rgba(124,58,237,0.4)',
              }}>Restart with New Resources</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
