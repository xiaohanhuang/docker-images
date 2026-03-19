/** @type {import('next').NextConfig} */
// NEXT_PUBLIC_* variables are baked into the JS bundle at build time and cannot
// be overridden via Kubernetes env vars.  The API URL is handled by the /api
// rewrite below (evaluated server-side at runtime), so the frontend never needs
// a real NEXT_PUBLIC_API_URL — it always calls /api/*.
// NEXT_PUBLIC_GRAFANA_URL is exposed via /api/config at runtime (see app/api/config/route.ts).
const API_BACKEND =
  process.env.NEXT_PUBLIC_API_URL ||
  'http://ml-platform-api.ml-platform-development.svc.cluster.local:8000';

const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  skipTrailingSlashRedirect: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${API_BACKEND}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
