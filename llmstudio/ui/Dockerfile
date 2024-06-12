
FROM oven/bun:canary-debian as base

WORKDIR /app

ENV NODE_ENV=production

FROM base as deps

COPY package.json ./
RUN bun install

FROM node:18 as build

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules

# COPY .env ./
COPY . .
RUN npm run build -- --no-lint

FROM base as final

USER bun

COPY --from=build /app/package.json ./package.json
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public

EXPOSE 3000

# CMD ["bun", "run", "start"]
