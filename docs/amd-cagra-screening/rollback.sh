#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(git rev-parse --show-toplevel)
git -C "$ROOT" apply -R "$ROOT/docs/amd-cagra-screening/niter50.patch"
