import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/ results and READMEs at build time, so every number on the page is generated, never typed.
export default defineConfig({
  base: '/llm-from-scratch/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
