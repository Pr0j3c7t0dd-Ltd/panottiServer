FROM node:20-alpine

WORKDIR /app

# Install build essentials
RUN apk add --no-cache python3 make g++

# Copy package files first
COPY package*.json ./

# Install ALL dependencies and keep them for build
RUN npm ci

# Copy project configuration files
COPY next.config.js tsconfig.json postcss.config.js tailwind.config.ts ./
COPY .eslintrc.json .prettierrc ./

# Copy source files
COPY src/ ./src/
COPY public/ ./public/

# Copy environment files if they exist
COPY .env* ./
COPY password-hash.txt ./

# Set environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=54790
ENV NODE_ENV=production

# Ensure node_modules/.bin is in PATH
ENV PATH /app/node_modules/.bin:$PATH

# Build the application
RUN npm run build

# Expose the port
EXPOSE 54790

# Start the application
CMD ["npm", "start"]
