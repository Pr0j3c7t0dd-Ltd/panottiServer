FROM node:20-alpine

WORKDIR /app

# Install dependencies first (better caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy configuration files
COPY postcss.config.js tailwind.config.ts tsconfig.json ./

# Copy the rest of the application
COPY . .

# Set environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=development
ENV PORT=54790

# Build the application
RUN npm run build

# Expose the port
EXPOSE 54790

# Start the application in development mode
CMD ["npm", "run", "dev"]
