
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  Box,
  BookOpen,
  Cpu,
  Database,
  DollarSign,
  GitBranch,
  Globe,
  Home,
  Layers,
  LayoutGrid,
  LineChart,
  MessageCircle,
  Monitor,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Users,
} from 'lucide-react';

/* ──── JupyterHub ingress base URL ──── */
const JUPYTERHUB_URL = import.meta.env.VITE_JUPYTERHUB_URL || '';

interface NavItem {
  name: string;
  href: string;
  icon: any;
  external?: boolean; // opens in new tab
}

interface NavSection {
  label: string;
  items: NavItem[];
}

function buildNavSections(username: string): NavSection[] {
  return [
    {
      label: 'Workspace',
      items: [
        {
          name: 'JupyterLab',
          href: `${JUPYTERHUB_URL}/user/${username}/lab`,
          icon: BookOpen,
          external: true,
        },
      ],
    },
    {
      label: 'Core',
      items: [
        { name: 'Home', href: '/', icon: Home },
        { name: 'Desks', href: '/desks', icon: Box },
        { name: 'Jobs & Pipelines', href: '/jobs', icon: GitBranch },
        { name: 'Components', href: '/components-lib', icon: Layers },
        { name: 'Recipes', href: '/recipes', icon: LayoutGrid },
      ],
    },
    {
      label: 'Intelligence',
      items: [
        { name: 'Experiments', href: '/experiments', icon: BarChart3 },
        { name: 'Model Registry', href: '/models', icon: Database },
        { name: 'Serving', href: '/serving', icon: Globe },
        { name: 'Data Lineage', href: '/data', icon: LineChart },
        { name: 'TensorBoard', href: '/tensorboard', icon: Monitor },
      ],
    },
    {
      label: 'Operations',
      items: [
        { name: 'Cost Center', href: '/cost', icon: DollarSign },
        { name: 'Ray Cluster', href: '/ray', icon: Activity },
        { name: 'Infrastructure', href: '/infrastructure', icon: Cpu },
        { name: 'Team Spaces', href: '/teams', icon: Users },
      ],
    },
    {
      label: '',
      items: [{ name: 'AI Assistant', href: '/chat', icon: MessageCircle }],
    },
  ];
}

/** Derive initials from a display name or username. */
function toInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
}

export function Navigation() {
  const location = useLocation();
  const pathname = location.pathname;
  const [collapsed, setCollapsed] = useState(false);
  const username = 'xiaohan'; // Hardcoded for now
  const navSections = buildNavSections(username);

  // Persist collapsed state
  useEffect(() => {
    const saved = localStorage.getItem('sidebar-collapsed');
    if (saved === 'true') setCollapsed(true);
  }, []);

  const toggle = () => {
    setCollapsed((prev) => {
      localStorage.setItem('sidebar-collapsed', String(!prev));
      return !prev;
    });
  };

  return (
    <nav
      className="app-sidebar"
      style={{
        width: collapsed ? 56 : undefined,
        minWidth: collapsed ? 56 : undefined,
        transition: 'width 0.2s ease, min-width 0.2s ease',
        overflow: 'hidden',
      }}
    >
      <div className="sidebar-header" style={{ justifyContent: collapsed ? 'center' : undefined }}>
        {!collapsed && (
          <>
            <div className="sidebar-logo">AD</div>
            <span className="sidebar-title">AI Desk</span>
          </>
        )}
        <button
          onClick={toggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--text-dimmed)',
            padding: 4,
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginLeft: collapsed ? 0 : 'auto',
          }}
        >
          {collapsed ? (
            <PanelLeftOpen style={{ width: 18, height: 18 }} />
          ) : (
            <PanelLeftClose style={{ width: 18, height: 18 }} />
          )}
        </button>
      </div>

      <div className="sidebar-nav">
        {navSections.map((section) => (
          <div key={section.label || 'extra'}>
            {section.label && !collapsed && (
              <div className="sidebar-section-label">{section.label}</div>
            )}
            {collapsed && section.label && (
              <div style={{ height: 8 }} />
            )}
            {section.items.map((item) => {
              const isActive =
                !item.external && (pathname === item.href ||
                (item.href !== '/' && pathname.startsWith(item.href)));
              const Icon = item.icon;

              if (item.external) {
                return (
                  <a
                    key={item.name}
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`sidebar-link`}
                    title={collapsed ? item.name : undefined}
                    style={collapsed ? { justifyContent: 'center', padding: '10px 0' } : undefined}
                  >
                    <Icon />
                    {!collapsed && <span>{item.name}</span>}
                  </a>
                );
              }

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`sidebar-link ${isActive ? 'active' : ''}`}
                  title={collapsed ? item.name : undefined}
                  style={collapsed ? { justifyContent: 'center', padding: '10px 0' } : undefined}
                >
                  <Icon />
                  {!collapsed && <span>{item.name}</span>}
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      {!collapsed && (
        <div className="sidebar-footer">
          <div className="sidebar-avatar">XH</div>
          <div className="sidebar-user-info">
            <div className="sidebar-user-name">Xiaohan Huang</div>
            <div className="sidebar-user-role">ML Engineer</div>
          </div>
          <Link to="/settings">
            <Settings
              style={{ width: 16, height: 16, color: 'var(--text-dimmed)', cursor: 'pointer' }}
            />
          </Link>
        </div>
      )}
      {collapsed && (
        <div className="sidebar-footer" style={{ justifyContent: 'center', padding: '12px 0' }}>
          <div className="sidebar-avatar" style={{ width: 28, height: 28, fontSize: 10 }}>XH</div>
        </div>
      )}
    </nav>
  );
}
