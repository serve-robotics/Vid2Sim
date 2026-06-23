#!/bin/bash
set -e

# Change to the directory where this script is located
cd "$(dirname "$0")"

echo "Building the Docker image..."
docker compose build
