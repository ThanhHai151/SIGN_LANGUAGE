import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const apiProxy = { '/api': 'http://127.0.0.1:8000' }

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  preview: { proxy: apiProxy },
  server: {
    proxy: apiProxy,
    watch: {
      ignored: ['backend', '.venv', '.venv-sstcn'].map((directory) =>
        fileURLToPath(new URL(`./${directory}`, import.meta.url)),
      ),
    },
  },
})
