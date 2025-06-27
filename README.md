# RAGRetriever
## 20 Newsgroups RAG System
This project implements a Retrieval-Augmented Generation (RAG) system for querying the 20 Newsgroups dataset using LangChain, Ollama, and PostgreSQL with pgvector. The system loads, processes, and embeds text documents into a vector store, enabling interactive Q&A about the dataset's topics.

## Features

Document Processing: Loads and splits 20 Newsgroups text files into ~682 chunks using RecursiveCharacterTextSplitter.
Vector Store: Stores embeddings in PostgreSQL with pgvector using the all-minilm model.
Query Processing: Uses the mistral model for generating answers.
Interactive Q&A: Supports interactive queries via terminal or non-interactive mode with a default query ("What are common topics in the 20 Newsgroups dataset?").
Dockerized Setup: Runs rag-app, ollama, and db services with Docker Compose.

## Prerequisites

- Docker and Docker Compose: Install Docker Desktop or Docker on Linux.
- System Resources: At least 16GB RAM (for mistral) and 4 CPU cores.
- 20 Newsgroups Dataset: Download from https://www.kaggle.com/datasets/crawford/20-newsgroups and place in ./data/20_newsgroups.

# Setup Instructions

## 1. Clone the Repository:
```sh
git clone <repository-url>
cd <repository-name>
```

## 2. Download the Dataset:

Download the 20 Newsgroups dataset from https://www.kaggle.com/datasets/crawford/20-newsgroups
Extract it to ./data/20_newsgroups, ensuring .txt files are in subdirectories (e.g., comp.graphics, sci.med).


## 3. Set Up Environment:

Create a .env file in the project root:
```sh
echo "DB_CONNECTION=postgresql://postgres:postgres@db:5432/rag_db" > .env
echo "OLLAMA_URL=http://ollama:11434" >> .env
echo "DEFAULT_QUERY=What are common topics in the 20 Newsgroups dataset?" >> .env
```

## 4. Build and Run with Docker Compose:
```sh
- docker compose down -v
- docker compose build
- docker compose up -d
```

## Usage

Interactive Mode:
```sh
docker compose up
```
Non-Interactive Mode:
```sh
docker compose exec rag-app python app/main.py
```

## Project Structure
```sh
├── data/20_newsgroups/        # Dataset directory
├── app/
│   ├── main.py                # Entry point for RAG pipeline
│   ├── vector_db.py           # Vector store creation
│   ├── document_processor.py  # Document loading and splitting
│   ├── rag_pipeline.py        # Query processing
│   ├── vector_manager.py      # Vector store management
│   ├── database.py            # Database initialization
│   ├── logging_config.py      # Logging setup
├── Dockerfile                 # rag-app Docker configuration
├── Dockerfile.ollama          # Ollama service configuration
├── docker-compose.yml         # Docker Compose configuration
├── entrypoint.sh              # Ollama service startup script
├── init.sql                   # Database initialization
├── schema.sql                 # Vector store schema
├── .env-example               # Environment variables
├── README.md                  # This file
```
