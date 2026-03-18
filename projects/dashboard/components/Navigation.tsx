'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  BarChart3,
  Box,
  Cpu,
  Database,
  GitBranch,
  Layers,
  LayoutDashboard,
} from 'lucide-react';

const navigation = [
  { name: 'Overview', href: '/overview', icon: LayoutDashboard },
  { name: 'Experiments', href: '/experiments', icon: BarChart3 },
  { name: 'Pipelines', href: '/pipelines', icon: GitBranch },
  { name: 'Ray', href: '/ray', icon: Activity },
  { name: 'Infrastructure', href: '/infrastructure', icon: Cpu },
  { name: 'Kubernetes', href: '/kubernetes', icon: Layers },
  { name: 'Models', href: '/models', icon: Database },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-col w-64 bg-gray-900 text-white">
      <div className="flex items-center h-16 px-6 border-b border-gray-700">
        <Box className="w-8 h-8 text-blue-400" />
        <h1 className="ml-3 text-xl font-bold">ML Platform</h1>
      </div>
      <div className="flex-1 px-3 py-4 overflow-auto">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`flex items-center px-3 py-2 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="ml-3">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="px-6 py-4 border-t border-gray-700">
        <p className="text-xs text-gray-400">ML Platform v0.1.0</p>
      </div>
    </nav>
  );
}
