/**
 * Config-driven widget renderer — renders Recharts charts from a WidgetSpec JSON.
 */
'use client';

import {
  LineChart, Line,
  BarChart, Bar,
  AreaChart, Area,
  PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const DEFAULT_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe',
  '#00c49f', '#ffbb28', '#ff8042', '#a4de6c', '#d0ed57',
];

interface WidgetSeries {
  name: string;
  dataKey: string;
  color?: string;
}

export interface WidgetSpec {
  type: 'line' | 'bar' | 'area' | 'pie' | 'stat' | 'table';
  title: string;
  description?: string;
  data?: Record<string, any>[];
  xAxisKey?: string;
  series?: WidgetSeries[];
  // Stat
  value?: string;
  unit?: string;
  trend?: number;
  // Table
  columns?: string[];
  rows?: Record<string, any>[];
  // Live
  live?: boolean;
  refreshQuery?: string;
}

interface WidgetRendererProps {
  spec: WidgetSpec;
  onPin?: (spec: WidgetSpec) => void;
  className?: string;
}

export default function WidgetRenderer({ spec, onPin, className = '' }: WidgetRendererProps) {
  return (
    <div className={`bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">{spec.title}</h3>
          {spec.description && (
            <p className="text-xs text-gray-500 mt-0.5">{spec.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {spec.live && (
            <span className="inline-flex items-center gap-1 text-xs text-green-600">
              <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
              Live
            </span>
          )}
          {onPin && (
            <button
              onClick={() => onPin(spec)}
              className="text-xs text-gray-400 hover:text-blue-600 transition-colors"
              title="Pin to workspace"
            >
              📌
            </button>
          )}
        </div>
      </div>
      <div className="px-4 pb-4">
        {renderWidget(spec)}
      </div>
    </div>
  );
}

function renderWidget(spec: WidgetSpec) {
  switch (spec.type) {
    case 'line': return <LineChartWidget spec={spec} />;
    case 'bar': return <BarChartWidget spec={spec} />;
    case 'area': return <AreaChartWidget spec={spec} />;
    case 'pie': return <PieChartWidget spec={spec} />;
    case 'stat': return <StatWidget spec={spec} />;
    case 'table': return <TableWidget spec={spec} />;
    default: return <p className="text-gray-500 text-sm">Unknown widget type: {spec.type}</p>;
  }
}

function LineChartWidget({ spec }: { spec: WidgetSpec }) {
  const series = spec.series || [];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={spec.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={spec.xAxisKey} tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {series.map((s, i) => (
          <Line
            key={s.dataKey}
            type="monotone"
            dataKey={s.dataKey}
            name={s.name}
            stroke={s.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function BarChartWidget({ spec }: { spec: WidgetSpec }) {
  const series = spec.series || [];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={spec.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={spec.xAxisKey} tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {series.map((s, i) => (
          <Bar
            key={s.dataKey}
            dataKey={s.dataKey}
            name={s.name}
            fill={s.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function AreaChartWidget({ spec }: { spec: WidgetSpec }) {
  const series = spec.series || [];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={spec.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={spec.xAxisKey} tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        {series.map((s, i) => (
          <Area
            key={s.dataKey}
            type="monotone"
            dataKey={s.dataKey}
            name={s.name}
            stroke={s.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
            fill={s.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
            fillOpacity={0.3}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

function PieChartWidget({ spec }: { spec: WidgetSpec }) {
  const data = spec.data || [];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          dataKey={spec.series?.[0]?.dataKey || 'value'}
          nameKey={spec.xAxisKey || 'name'}
          cx="50%"
          cy="50%"
          outerRadius={100}
          label={({ name, percent }: any) => `${name ?? ''} ${((percent ?? 0) * 100).toFixed(0)}%`}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={DEFAULT_COLORS[i % DEFAULT_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}

function StatWidget({ spec }: { spec: WidgetSpec }) {
  return (
    <div className="flex items-center gap-4 py-4">
      <div>
        <div className="text-3xl font-bold text-gray-900">
          {spec.value}
          {spec.unit && <span className="text-lg text-gray-500 ml-1">{spec.unit}</span>}
        </div>
        {spec.trend !== undefined && spec.trend !== null && (
          <div className={`text-sm mt-1 ${spec.trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {spec.trend >= 0 ? '↑' : '↓'} {Math.abs(spec.trend).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
}

function TableWidget({ spec }: { spec: WidgetSpec }) {
  const columns = spec.columns || [];
  const rows = spec.rows || [];
  return (
    <div className="overflow-x-auto max-h-80">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((col) => (
              <th key={col} className="px-3 py-2 text-left font-medium text-gray-600">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-gray-50">
              {columns.map((col) => (
                <td key={col} className="px-3 py-2 text-gray-800">
                  {String(row[col] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
