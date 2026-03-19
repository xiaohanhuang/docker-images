import { formatDistanceToNow } from 'date-fns';

export function formatDate(date: string | Date): string {
  return formatDistanceToNow(new Date(date), { addSuffix: true });
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export function getStatusColor(status: string): string {
  const statusColors: Record<string, string> = {
    Running: 'bg-green-100 text-green-800',
    Succeeded: 'bg-blue-100 text-blue-800',
    Failed: 'bg-red-100 text-red-800',
    Pending: 'bg-yellow-100 text-yellow-800',
    Unknown: 'bg-gray-100 text-gray-800',
  };
  return statusColors[status] || statusColors.Unknown;
}
