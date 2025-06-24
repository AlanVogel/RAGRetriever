#!/bin/bash
set -e

# Pull models in background or handle failures without crashing
ollama pull llama4 || true
ollama pull mxbai-embed-large || true

# Start the Ollama server
exec ollama serve