import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Providers } from './providers';
import { Navigation } from '@/components/Navigation';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ML Platform Dashboard',
  description: 'Unified dashboard for ML training platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <div className="flex h-screen">
            <Navigation />
            <main className="flex-1 overflow-auto bg-gray-50">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
