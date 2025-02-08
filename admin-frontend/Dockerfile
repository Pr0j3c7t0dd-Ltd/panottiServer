FROM node:20-alpine

WORKDIR /app

# Install build essentials
RUN apk add --no-cache python3 make g++

# Install ALL dependencies (including dev dependencies)
COPY package*.json ./
COPY package-lock.json ./
# Temporarily set NODE_ENV to development for full installation
ENV NODE_ENV=development
RUN npm ci
ENV NODE_ENV=production

# Copy configuration files first
COPY postcss.config.js postcss.config.mjs tailwind.config.ts tsconfig.json next.config.js .eslintrc.json .prettierrc ./

# Copy sensitive files
COPY .env.local ./.env.local
COPY password-hash.txt ./password-hash.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=54790

# Ensure node_modules/.bin is in PATH
ENV PATH /app/node_modules/.bin:$PATH

# Build the application
RUN npm run build

# Expose the port
EXPOSE 54790

# Start the application
CMD ["npm", "start"]
