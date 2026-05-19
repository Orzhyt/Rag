import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiTarget = `http://localhost:${process.env.API_PORT || '8000'}`

export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.HOST || '0.0.0.0',
    port: parseInt(process.env.PORT || '3000'),
    proxy: {
      '/health': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
