import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const certPath = path.resolve(dirname, '../certs/sign-link-dev.pfx');
const hasDevCertificate = fs.existsSync(certPath);

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: hasDevCertificate
      ? {
          pfx: fs.readFileSync(certPath),
          passphrase: process.env.VITE_HTTPS_PASSPHRASE || 'signlink-dev'
        }
      : undefined,
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:5001',
        changeOrigin: true,
        secure: false
      },
      '/socket.io': {
        target: process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:5001',
        changeOrigin: true,
        secure: false,
        ws: true
      },
      '/landmark-api': {
        target: process.env.VITE_LANDMARK_PROXY_TARGET || 'http://172.16.224.122:8001',
        changeOrigin: true,
        secure: false,
        rewrite: (requestPath) => requestPath.replace(/^\/landmark-api/, ''),
        headers: {
          'bypass-tunnel-reminder': 'true'
        }
      }
    }
  }
});
