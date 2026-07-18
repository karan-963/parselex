/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use Turbopack (default in Next.js 16). Empty config silences the
  // "webpack config present but no turbopack config" warning.
  turbopack: {
    root: __dirname,
  },
};

module.exports = nextConfig;
