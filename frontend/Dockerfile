# ============================================================
# Stage 1: Dev — used by docker-compose with volume mounts
# ============================================================
FROM node:20-alpine AS dev

WORKDIR /app

COPY package*.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]

# ============================================================
# Stage 2: Build — compile Vite production bundle
# ============================================================
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci

ARG VITE_BACKEND_URL=/api
COPY . .
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

RUN npm run build

# ============================================================
# Stage 3: Production — nginx serves static assets
# ============================================================
FROM nginx:alpine AS production

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]