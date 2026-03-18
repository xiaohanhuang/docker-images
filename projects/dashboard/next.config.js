/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://ml-platform-api.ml-platform-development.svc.cluster.local:8000',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://ml-platform-api.ml-platform-development.svc.cluster.local:8000'}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
