import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En desarrollo la API local (uvicorn web.servidor:app --port 8080) sirve /api, /resultados y /datos.
// En AWS, CloudFront sirve la página y los resultados; la API es la Function URL (VITE_API_URL).
const api = process.env.API_LOCAL ?? "http://localhost:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": api, "/resultados": api, "/datos": api },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
