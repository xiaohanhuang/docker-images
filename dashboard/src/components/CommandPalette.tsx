
import { Command } from 'cmdk';
import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import {
  Activity, BarChart3, Box, Cpu, Database, DollarSign,
  GitBranch, Globe, Home, Layers, LayoutGrid,
  LineChart, Network, Play, Search, Settings, Square,
  Users, Zap, FlaskConical, Terminal,
} from 'lucide-react';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  const runCommand = (command: () => void) => {
    setOpen(false);
    command();
  };

  if (!open) return null;

  return (
    <div className="cmdk-overlay" onClick={() => setOpen(false)}>
      <div className="cmdk-dialog" onClick={(e) => e.stopPropagation()}>
        <Command>
          <div className="cmdk-input-wrapper">
            <Search />
            <Command.Input placeholder="Type a command or search... (e.g. 'launch recipe llm-rlhf medium')" />
          </div>
          <Command.List>
            <Command.Empty>No results found.</Command.Empty>

            <Command.Group heading="Navigation">
              <Command.Item onSelect={() => runCommand(() => navigate('/'))}>
                <Home /> <span>Home Dashboard</span> <span className="cmdk-shortcut">⌘1</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/desks'))}>
                <Box /> <span>My Desks</span> <span className="cmdk-shortcut">⌘2</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/jobs'))}>
                <GitBranch /> <span>Jobs & Pipelines</span> <span className="cmdk-shortcut">⌘3</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/recipes'))}>
                <LayoutGrid /> <span>Recipe Catalog</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/experiments'))}>
                <BarChart3 /> <span>Experiments</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/models'))}>
                <Database /> <span>Model Registry</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/serving'))}>
                <Globe /> <span>Serving & Endpoints</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/data'))}>
                <LineChart /> <span>Data Lineage</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/cost'))}>
                <DollarSign /> <span>Cost Center</span> <span className="cmdk-shortcut">⌘4</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/components-lib'))}>
                <Layers /> <span>Component Library</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/ray'))}>
                <Activity /> <span>Ray Cluster</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/infrastructure'))}>
                <Cpu /> <span>Infrastructure</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/teams'))}>
                <Users /> <span>Team Spaces</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="Actions">
              <Command.Item onSelect={() => runCommand(() => navigate('/desks'))}>
                <Zap /> <span>Launch New Desk</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/recipes/llm-rlhf'))}>
                <Play /> <span>Launch Recipe: llm-rlhf</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/recipes/text2sql'))}>
                <Play /> <span>Launch Recipe: text2sql</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/recipes/sdxl-lora'))}>
                <Play /> <span>Launch Recipe: sdxl-lora</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/experiments'))}>
                <FlaskConical /> <span>Compare Experiment Runs</span>
              </Command.Item>
              <Command.Item keywords={['scale', 'gpu', 'a100', 'h100', 't4']}>
                <Cpu /> <span>Scale Desk GPU</span>
                <span className="cmdk-shortcut" style={{ fontSize: 10, color: 'var(--text-dimmed)' }}>requires desk</span>
              </Command.Item>
              <Command.Item keywords={['stop', 'kill', 'terminate']}>
                <Square /> <span>Stop Idle Desks</span>
                <span className="cmdk-shortcut" style={{ fontSize: 10, color: 'var(--error)' }}>destructive</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="Quick Access">
              <Command.Item onSelect={() => runCommand(() => {
                window.open('http://k8s-jupyter-jupyterh-827e6a6320-482154231.us-west-2.elb.amazonaws.com/user/xiaohan/vscode/', '_blank');
              })}>
                <Terminal /> <span>Open VS Code (code-server)</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => {
                window.open('http://k8s-jupyter-jupyterh-827e6a6320-482154231.us-west-2.elb.amazonaws.com/user/xiaohan/lab', '_blank');
              })}>
                <Box /> <span>Open JupyterLab</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/chat'))}>
                <Network /> <span>AI Assistant</span>
              </Command.Item>
            </Command.Group>

            <Command.Group heading="Settings">
              <Command.Item onSelect={() => runCommand(() => navigate('/settings'))}>
                <Settings /> <span>Settings & Preferences</span>
              </Command.Item>
              <Command.Item onSelect={() => runCommand(() => navigate('/chat'))}>
                <Network /> <span>AI Assistant</span>
              </Command.Item>
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
