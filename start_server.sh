#!/bin/bash

set -e

pids=()

cleanup() {
  echo "Cleaning up..."
  kill "${pids[@]}" 2>/dev/null || true
  wait
}

trap cleanup SIGINT SIGTERM EXIT

python -m http.server 5001 &
pids+=($!)

python annotation_tool/app.py &
pids+=($!)

wait
