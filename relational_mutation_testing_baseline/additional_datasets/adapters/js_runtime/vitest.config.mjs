import { defineConfig } from 'vitest/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.AD_WORKSPACE || process.cwd();

// The corpus is NOT uniform about where a holdout test expects its sources, so the adapter
// detects the layout per fixture and passes it in AD_SRC_PREFIX:
//   ''    — sources flat at the workspace root; the test imports `./app/index.js`
//   'src' — sources under src/;                the test imports `./src/app/route.js`
// `@` must follow the same prefix, because `@/app/api/utils/sql` is relative to the SOURCE
// root, not the workspace root. Measured over the 18 pure-logic fixtures: 7 flat,
// 4 src-prefixed, 7 unusable for unrelated reasons. Guessing one layout for all gives
// "Failed to resolve import" on whichever group you guessed against.
const srcRoot = path.join(root, process.env.AD_SRC_PREFIX || '');

export default defineConfig({
  root,
  // The automatic JSX runtime. Fixture components and holdout tests use JSX without
  // importing React, so the classic runtime fails every case with
  // "ReferenceError: React is not defined".
  esbuild: { jsx: 'automatic' },
  resolve: { alias: { '@': srcRoot } },
  test: {
    environment: 'jsdom',
    // `globals: true` is required, not a convenience: several holdout suites call
    // `beforeEach`/`vi` without importing them from 'vitest'. With globals off,
    // dijkstra_shortest_path dies with "beforeEach is not defined" and reports zero cases.
    globals: true,
    include: ['*.test.js', '*.test.jsx'],
    setupFiles: [path.join(here, 'vitest.setup.mjs')],
  },
});
