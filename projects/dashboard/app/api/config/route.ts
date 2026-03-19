import { NextResponse } from 'next/server';

/**
 * Runtime config endpoint — returns environment-specific URLs that can't be
 * baked in at build time (e.g. Grafana URL varies per deployment).
 * The frontend fetches this once on startup instead of relying on NEXT_PUBLIC_*
 * variables, which are substituted at build time and can't be overridden by
 * Kubernetes env vars at runtime.
 */
export function GET() {
  return NextResponse.json({
    grafanaUrl: process.env.GRAFANA_URL || 'http://grafana.ml-platform.internal',
  });
}
