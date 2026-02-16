import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 8501,
    allowedHosts: ["chatbotknds.mini.pw.edu.pl"],
    proxy: {
      "/api": {
        target: "http://api:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
    hmr: {
      protocol: "wss",
      host: "chatbotknds.mini.pw.edu.pl",
      clientPort: 443,
    },
  },
});
