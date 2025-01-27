/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    deviceSizes: [64, 256], // Only the sizes we actually use
    imageSizes: [], // Disable default image sizes
    formats: ['image/webp'],
    disableStaticImages: true
  },
  experimental: {
    cpus: 1,
    workerThreads: false,
    webpackBuildWorker: false,
    serverMinification: false,
    serverSourceMaps: false,
    optimizeServerReact: false,
    optimizeCss: false
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.optimization = {
        ...config.optimization,
        minimize: false
      }
    }
    return config
  },
  staticPageGenerationTimeout: 300,
  generateEtags: false,
  compress: false,
  poweredByHeader: false,
  productionBrowserSourceMaps: false,
  typescript: {
    ignoreBuildErrors: true
  },
  eslint: {
    ignoreDuringBuilds: true
  }
};

module.exports = nextConfig;
