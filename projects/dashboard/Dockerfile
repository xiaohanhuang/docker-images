# ============================================================
# AI Desk Platform — Dashboard (Vite SPA)
# Public builder fetching proprietary source from ECR
# ============================================================

FROM 805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/source:latest AS source

# ---- Stage 1: Dependencies ----
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY --from=source /src/projects/dashboard/package.json /src/projects/dashboard/package-lock.json* ./
RUN npm ci

# ---- Stage 2: Build ----
FROM node:20-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY --from=source /src/projects/dashboard/ .

# Vite bakes env vars prefixed with VITE_ at build time
ENV VITE_API_URL=http://ml-platform-api.ml-platform-development.svc.cluster.local:8000
RUN npm run build

# ---- Stage 3: Production (nginx) ----
FROM nginx:1.27-alpine AS runner

# Copy built static assets
COPY --from=builder /app/dist /usr/share/nginx/html

# SPA fallback: redirect all routes to index.html
RUN printf 'server {\n\
    listen 3000;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location /api/ {\n\
        proxy_pass http://ml-platform-api.ml-platform-development.svc.cluster.local:8000;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
    }\n\
    location /desk-proxy/ {\n\
        proxy_pass http://ml-platform-api.ml-platform-development.svc.cluster.local:8000;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
    }\n\
    location /desk-marimo/ {\n\
        proxy_pass http://ml-platform-api.ml-platform-development.svc.cluster.local:8000;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
    }\n\
    location /desk-jupyter/ {\n\
        proxy_pass http://ml-platform-api.ml-platform-development.svc.cluster.local:8000;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
    }\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
