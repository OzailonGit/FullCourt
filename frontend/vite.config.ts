import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the Python API runs separately (uvicorn on port 8000); requests to /api are
// forwarded there so the browser sees a single origin, as it does when FastAPI serves the build.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
