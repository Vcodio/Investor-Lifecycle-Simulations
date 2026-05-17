#!/usr/bin/env bash
# Wrapper: run from repo root so `./build_cython.sh` matches app hints.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/misc/build_cython.sh" "$@"
