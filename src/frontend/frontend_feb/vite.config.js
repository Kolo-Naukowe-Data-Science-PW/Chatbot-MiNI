import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // # dodane: proxy dla połączenia frontend -> API w Dockerze
  server: {
  // # dodane: dozwolone hosty dla Vite na VM
  allowedHosts: ["chatbotknds.mini.pw.edu.pl"],
  proxy: {
    "/api": {
      target: "http://api:8000",
      changeOrigin: true,
      },
    },
  },

})
