import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.js'],
    globals: true, // describe/test/expect available without imports (Jest parity)
    testTimeout: 30000,
    environment: 'node',
  },
});
