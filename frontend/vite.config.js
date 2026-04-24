import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  // Load root .env (one level up from frontend/)
  const env = loadEnv("development", "../", "");

  const authHeader = env.API_TOKEN
    ? { Authorization: `Bearer ${env.API_TOKEN}` }
    : {};

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/query": {
          target: "http://localhost:8000",
          changeOrigin: true,
          headers: authHeader,
        },
        "/health": {
          target: "http://localhost:8000",
          changeOrigin: true,
          headers: authHeader,
        },
      },
    },
  };
});
