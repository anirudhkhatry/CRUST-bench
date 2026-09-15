#!/usr/bin/env bash
# Install the shared vitest runtime once; every fixture run reuses it.
set -euo pipefail
cd "$(dirname "$0")/../additional_datasets/adapters/js_runtime"
npm install --no-audit --no-fund
./node_modules/.bin/vitest --version
