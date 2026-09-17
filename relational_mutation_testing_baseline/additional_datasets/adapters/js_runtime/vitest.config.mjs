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
  resolve: {
    alias: {
      '@': srcRoot,
      // `react-native` is aliased ONLY so Vite can RESOLVE the specifier. The React
      // Native fixtures' own holdouts already `vi.mock('react-native', factory)` with
      // their own DOM stand-ins, so the mock -- not react-native-web -- defines the
      // behaviour under test. Vite's import-analysis resolves the source file's imports
      // at transform time, before any mock applies, so without something on disk to point
      // at, the suite dies with "Failed to resolve import" and contributes zero cases
      // while the mock it shipped goes unused.
      'react-native': 'react-native-web',
      // `lucide-react-native` exports the same icon names as `lucide-react`, and the
      // fixtures importing it render through their own `vi.mock('react-native')` stand-ins,
      // so the DOM build is a faithful substitute for resolution purposes. ALIASING RATHER
      // THAN INSTALLING IS DELIBERATE: the real package will not install next to
      // react-native-web without `--legacy-peer-deps`, and that flag re-resolves the whole
      // tree and removes @testing-library/dom -- measured, it drops 10 packages and every
      // React fixture then reports zero cases. An alias cannot touch the tree.
      // `react-native-safe-area-context` has the same conflict and gets a local shim.
      'lucide-react-native': 'lucide-react',
      'react-native-safe-area-context': path.join(here, 'rn_shims/safe-area-context.mjs'),
    },
  },
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
