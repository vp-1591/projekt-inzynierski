// Backend URL — in production (behind nginx reverse proxy), set VITE_BACKEND_URL=/api at build time.
// In development, falls back to direct localhost connection.
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';