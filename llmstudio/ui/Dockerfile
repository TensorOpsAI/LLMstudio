FROM oven/bun:1 as base

WORKDIR /app

ENV NODE_ENV=production

FROM base as install

COPY package.json bun.lockb ./
RUN bun install

FROM node:18 as build

ARG NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST
ARG NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT
ARG NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST
ARG NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT

ENV NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST=$NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST
ENV NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT=$NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT
ENV NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST=$NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST
ENV NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT=$NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT

WORKDIR /app

COPY --from=install /app/node_modules ./node_modules
COPY . .
RUN npm run build -- --no-lint

FROM base as release

USER bun

COPY --from=build /app/package.json ./package.json
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public

EXPOSE 3000

# CMD ["bun", "run", "start"]