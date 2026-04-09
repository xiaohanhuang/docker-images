
import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Save, Monitor, Keyboard, DollarSign, Bell, Shield, Eye, Clock, Loader2 } from 'lucide-react';
import { fetchSettings, api } from '@/lib/api';
import { useToast } from '@/components/Toast';

export default function SettingsPage() {
  const { toast } = useToast();
  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => fetchSettings(),
    staleTime: 60_000,
    retry: false,
  });

  const [s, setS] = useState({
    default_namespace: 'ml-team',
    default_gpu_type: 'A100',
    default_recipe_profile: 'Medium',
    editor_font_size: 14,
    editor_key_bindings: 'Default',
    editor_minimap: true,
    editor_word_wrap: true,
    burn_rate_display: 'Badge',
    budget_alerts_enabled: true,
    weekly_report_email: true,
    budget_limit_monthly: 50000,
    idle_timeout_minutes: 120,
    auto_suspend_action: 'Suspend',
    ghost_desk_protection: true,
    reduce_motion: false,
    high_contrast: false,
    notification_job_complete: true,
    notification_idle_gpu: true,
    notification_budget_threshold: true,
    notification_budget_percent: 80,
    ...settings,
  });

  const update = (key: string, value: any) => setS((prev: Record<string, any>) => ({ ...prev, [key]: value }));

  const saveMutation = useMutation({
    mutationFn: () => api.settings.update(s),
    onSuccess: () => toast('Settings saved successfully', 'success'),
    onError: (err: any) => toast(`Failed to save: ${err?.message || 'Unknown error'}`, 'error'),
  });

  function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 0', borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}>
        <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>{label}</span>
        {children}
      </div>
    );
  }

  function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
    return (
      <button onClick={() => onChange(!value)} style={{
        width: 40, height: 22, borderRadius: 11, border: 'none', cursor: 'pointer',
        background: value ? 'var(--accent-primary)' : 'rgba(255,255,255,0.12)',
        position: 'relative', transition: 'background 0.2s',
      }}>
        <div style={{
          width: 16, height: 16, borderRadius: '50%', background: '#fff',
          position: 'absolute', top: 3, left: value ? 21 : 3, transition: 'left 0.2s',
        }} />
      </button>
    );
  }

  function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
    return (
      <select value={value} onChange={e => onChange(e.target.value)}
        style={{
          padding: '6px 10px', fontSize: 13, borderRadius: 'var(--radius-sm)',
          background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
          color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none',
        }}
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>Settings & Preferences</h1>
          <p>Configure your global defaults, editor, notifications, and accessibility</p>
        </div>
        <button className="btn btn-primary" onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
          {saveMutation.isPending ? (
            <><Loader2 style={{ width: 14, height: 14, animation: 'spin 1s linear infinite' }} /> Saving...</>
          ) : (
            <><Save style={{ width: 14, height: 14 }} /> Save All</>
          )}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Defaults */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Shield style={{ width: 16, height: 16 }} /> Defaults
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Default Namespace">
              <Select value={s.default_namespace} options={['ml-team', 'research', 'production', 'staging']}
                onChange={v => update('default_namespace', v)} />
            </SettingRow>
            <SettingRow label="Default GPU Type">
              <Select value={s.default_gpu_type} options={['T4', 'V100', 'A100', 'H100']}
                onChange={v => update('default_gpu_type', v)} />
            </SettingRow>
            <SettingRow label="Default Recipe Profile">
              <Select value={s.default_recipe_profile} options={['Small', 'Medium', 'Large']}
                onChange={v => update('default_recipe_profile', v)} />
            </SettingRow>
          </div>
        </div>

        {/* Editor Preferences */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Monitor style={{ width: 16, height: 16 }} /> Editor
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Font Size">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input type="range" min={11} max={20} value={s.editor_font_size}
                  onChange={e => update('editor_font_size', Number(e.target.value))}
                  style={{ width: 80, accentColor: 'var(--accent-primary)' }}
                />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, minWidth: 28, textAlign: 'right' }}>{s.editor_font_size}px</span>
              </div>
            </SettingRow>
            <SettingRow label="Key Bindings">
              <Select value={s.editor_key_bindings} options={['Default', 'Vim', 'Emacs']}
                onChange={v => update('editor_key_bindings', v)} />
            </SettingRow>
            <SettingRow label="Minimap">
              <Toggle value={s.editor_minimap} onChange={v => update('editor_minimap', v)} />
            </SettingRow>
            <SettingRow label="Word Wrap">
              <Toggle value={s.editor_word_wrap} onChange={v => update('editor_word_wrap', v)} />
            </SettingRow>
            <SettingRow label="Theme">
              <span style={{ fontSize: 13, color: 'var(--text-dimmed)', fontStyle: 'italic' }}>Dark (enforced)</span>
            </SettingRow>
          </div>
        </div>

        {/* Cost Preferences */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <DollarSign style={{ width: 16, height: 16 }} /> Cost Preferences
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Burn Rate Display">
              <Select value={s.burn_rate_display} options={['Badge', 'Hidden', 'Copilot Only']}
                onChange={v => update('burn_rate_display', v)} />
            </SettingRow>
            <SettingRow label="Budget Alerts">
              <Toggle value={s.budget_alerts_enabled} onChange={v => update('budget_alerts_enabled', v)} />
            </SettingRow>
            <SettingRow label="Weekly Report Email">
              <Toggle value={s.weekly_report_email} onChange={v => update('weekly_report_email', v)} />
            </SettingRow>
            <SettingRow label="Monthly Budget Limit">
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                $<input type="number" value={s.budget_limit_monthly} onChange={e => update('budget_limit_monthly', Number(e.target.value))}
                  style={{
                    width: 80, padding: '4px 6px', textAlign: 'right', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 13, outline: 'none',
                  }}
                />
              </div>
            </SettingRow>
          </div>
        </div>

        {/* Notifications */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Bell style={{ width: 16, height: 16 }} /> Notifications
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Job Completion">
              <Toggle value={s.notification_job_complete} onChange={v => update('notification_job_complete', v)} />
            </SettingRow>
            <SettingRow label="Idle GPU Alerts">
              <Toggle value={s.notification_idle_gpu} onChange={v => update('notification_idle_gpu', v)} />
            </SettingRow>
            <SettingRow label="Budget Threshold Alert">
              <Toggle value={s.notification_budget_threshold} onChange={v => update('notification_budget_threshold', v)} />
            </SettingRow>
            <SettingRow label="Budget Alert at %">
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                <input type="number" value={s.notification_budget_percent} min={50} max={100}
                  onChange={e => update('notification_budget_percent', Number(e.target.value))}
                  style={{
                    width: 50, padding: '4px 6px', textAlign: 'center', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 13, outline: 'none',
                  }}
                />%
              </div>
            </SettingRow>
          </div>
        </div>

        {/* Auto-Suspend */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Clock style={{ width: 16, height: 16 }} /> Auto-Suspend Policy
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Idle Timeout">
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                <input type="number" value={s.idle_timeout_minutes} min={30} max={720}
                  onChange={e => update('idle_timeout_minutes', Number(e.target.value))}
                  style={{
                    width: 60, padding: '4px 6px', textAlign: 'center', borderRadius: 4,
                    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 13, outline: 'none',
                  }}
                /> min
              </div>
            </SettingRow>
            <SettingRow label="Suspend Action">
              <Select value={s.auto_suspend_action} options={['Suspend', 'Stop', 'Hibernate']}
                onChange={v => update('auto_suspend_action', v)} />
            </SettingRow>
            <SettingRow label="Ghost Desk Protection">
              <Toggle value={s.ghost_desk_protection} onChange={v => update('ghost_desk_protection', v)} />
            </SettingRow>
            <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-dimmed)', lineHeight: 1.5 }}>
              When enabled, desks with &lt;5% GPU utilization for the configured timeout are automatically suspended, preserving EFS disk state.
            </div>
          </div>
        </div>

        {/* Accessibility */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Eye style={{ width: 16, height: 16 }} /> Accessibility
            </span>
          </div>
          <div className="card-body">
            <SettingRow label="Reduce Motion">
              <Toggle value={s.reduce_motion} onChange={v => update('reduce_motion', v)} />
            </SettingRow>
            <SettingRow label="High Contrast Mode">
              <Toggle value={s.high_contrast} onChange={v => update('high_contrast', v)} />
            </SettingRow>
            <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-dimmed)', lineHeight: 1.5 }}>
              High contrast mode increases text contrast and border visibility. The platform enforces dark-mode-only for WCAG 2.2 compliance.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
