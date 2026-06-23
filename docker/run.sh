#!/bin/bash
set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

echo "Starting and attaching to the Docker container..."
docker compose run --rm vid2sim bash
