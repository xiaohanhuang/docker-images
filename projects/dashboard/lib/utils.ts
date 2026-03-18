export function formatDate(dateString: string | null | undefined): string {
  if (!dateString) return '—';
  const date = new Date(dateString);
  if (isNaN(date.getTime())) return '—';
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatCurrency(amount: number | null | undefined): string {
  if (amount == null) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(amount);
}

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-green-100 text-green-800',
  succeeded: 'bg-blue-100 text-blue-800',
  failed: 'bg-red-100 text-red-800',
  pending: 'bg-yellow-100 text-yellow-800',
  unknown: 'bg-gray-100 text-gray-800',
};

export function getStatusColor(status: string): string {
  return STATUS_COLORS[status.toLowerCase()] ?? STATUS_COLORS.unknown;
}
