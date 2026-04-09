import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Navigation } from './components/Navigation';
import { ChatSidebar } from './components/ChatSidebar';
import { ChatProvider } from './lib/ChatContext';
import { ToastProvider } from './components/Toast';
import TerminalWidget from './components/TerminalWidget';
import MockBanner from './components/MockBanner';

// Page imports — every page from the old Next.js app/ directory
import Home from './pages/page';
import CostCenter from './pages/cost/page';
import Desks from './pages/desks/page';
import DeskIDEPage from './pages/desks/[id]/page';
import ComponentsLib from './pages/components-lib/page';
import Jobs from './pages/jobs/page';
import JobDetail from './pages/jobs/[id]/page';
import Experiments from './pages/experiments/page';
import Models from './pages/models/page';
import ModelDetail from './pages/models/[id]/page';
import Recipes from './pages/recipes/page';
import RecipeDetail from './pages/recipes/[id]/page';
import Serving from './pages/serving/page';
import Chat from './pages/chat/page';
import Ray from './pages/ray/page';
import Infrastructure from './pages/infrastructure/page';
import Teams from './pages/teams/page';
import Data from './pages/data/page';
import Settings from './pages/settings/page';
import TensorBoard from './pages/tensorboard/page';

function Layout() {
  const [terminalOpen, setTerminalOpen] = useState(false);

  return (
    <div className="app-layout">
      <Navigation />
      <main className="app-main relative">
        <MockBanner />
        <div className="page-wrapper">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/cost" element={<CostCenter />} />
            <Route path="/desks" element={<Desks />} />
            <Route path="/desks/:id" element={<DeskIDEPage />} />
            <Route path="/components-lib" element={<ComponentsLib />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="/jobs/:id" element={<JobDetail />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/models" element={<Models />} />
            <Route path="/models/:id" element={<ModelDetail />} />
            <Route path="/recipes" element={<Recipes />} />
            <Route path="/recipes/:id" element={<RecipeDetail />} />
            <Route path="/serving" element={<Serving />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/ray" element={<Ray />} />
            <Route path="/infrastructure" element={<Infrastructure />} />
            <Route path="/teams" element={<Teams />} />
            <Route path="/data" element={<Data />} />
            <Route path="/tensorboard" element={<TensorBoard />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
        <ChatSidebar />
      </main>

      <button
        className={`terminal-toggle ${terminalOpen ? 'active' : ''}`}
        onClick={() => setTerminalOpen(!terminalOpen)}
        title="Toggle Terminal"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="4 17 10 11 4 5"></polyline>
          <line x1="12" y1="19" x2="20" y2="19"></line>
        </svg>
      </button>

      {terminalOpen && (
        <div className="terminal-overlay">
          <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '4px 8px' }}>
            <button onClick={() => setTerminalOpen(false)} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: 16 }}>✕</button>
          </div>
          <TerminalWidget deskId="default" />
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <ChatProvider>
      <ToastProvider>
        <Layout />
      </ToastProvider>
    </ChatProvider>
  );
}
