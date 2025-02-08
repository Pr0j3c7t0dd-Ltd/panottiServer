FROM node:20-alpine

WORKDIR /app

# Install ALL dependencies for build (including dev dependencies)
COPY package.json package-lock.json ./
RUN npm ci --include=dev

# Copy configuration files
COPY postcss.config.js postcss.config.mjs tailwind.config.ts tsconfig.json next.config.js .eslintrc.json .prettierrc ./

# Copy sensitive files
COPY .env.local ./.env.local
COPY password-hash.txt ./password-hash.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
ENV PORT=54790

# Build the application
RUN npm run build

# Remove dev dependencies after build
RUN npm ci --omit=dev

# Expose the port
EXPOSE 54790

# Start the application
CMD ["npm", "start"]
