#!/bin/bash
set -e

echo "Starting Ollama server in the background..."
ollama serve &

echo "Waiting for Ollama server to become available..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 1
done

echo "Pulling models..."
ollama pull mistral || echo "Failed to pull mistral"
ollama pull all-minilm || echo "Failed to pull all-minilm"

echo "Waiting for models to be available..."
for MODEL in mistral all-minilm; do
  echo "Waiting for $MODEL to be available..."
  until curl -s http://localhost:11434/api/show -d "{\"name\": \"$MODEL\"}" | grep -q "\"modelfile\""; do
    sleep 1
  done
done

echo "All models are ready."
wait