
import { IS_MOCK } from '@/lib/api';

/**
 * Tiny floating banner shown only when VITE_MOCK=true.
 * Reminds developers they are seeing fixture data.
 */
export default function MockBanner() {
  if (!IS_MOCK) return null;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 12,
        right: 12,
        zIndex: 9999,
        background: '#f59e0b',
        color: '#000',
        padding: '4px 12px',
        borderRadius: 6,
        fontSize: 12,
        fontWeight: 600,
        boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
        pointerEvents: 'none',
      }}
    >
      MOCK MODE
    </div>
  );
}
