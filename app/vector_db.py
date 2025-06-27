import os
import requests
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from database import initialize_database
from document_processor import DocumentProcessor
from vector_manager import VectorStoreManager
from logging_config import setup_logging
from sqlalchemy import create_engine, text

logger = setup_logging()
load_dotenv()

def initialize_components(data_dir = "./data/20_newsgroups"):
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    embedding_model = "all-minilm"
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

    logger.info("Checking Ollama model availability")
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code != 200 or embedding_model not in response.text:
            logger.error(f"Model {embedding_model} not found at {ollama_url}")
            return None, None, None
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return None, None, None
    
    logger.info("Initializing database")
    Session = initialize_database(connection_string)

    logger.info(f"Initializing embeddings with model {embedding_model}")

    try:
        embeddings = OllamaEmbeddings(
            model = embedding_model,
            base_url = ollama_url,
            num_ctx=4096
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return None, None, None
    
    logger.info(f"Initializing document processor")
    processor = DocumentProcessor(data_dir)

    return Session, embeddings, processor

def create_vector_db(batch_size=100, max_batches=300):
    session_factory, embeddings, processor = initialize_components()
    if not session_factory or not embeddings or not processor:
        return
    
    #checker
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    with create_engine(connection_string).connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = 'newsgroups_vectors')"))
        existing_doc_count = result.scalar() or 0
    if existing_doc_count >= 1000:  # Assume complete if many embeddings exist
        logger.info(f"Vector store already contains {existing_doc_count} embeddings, skipping creation")
        return session_factory, embeddings, VectorStoreManager(session_factory, embeddings)

    documents = processor.load_documents()
    if not documents:
        logger.warning("No documents found in the specified directory")
        return
    logger.info(f"Total documents loaded: {len(documents)}")

    texts = processor.split_documents(documents)
    logger.info("Creating vector store")
    if len(texts) > 5000:
        logger.warning(f"High chunk count ({len(texts)}), consider increasing chunk_size")

    vector_store_manager = VectorStoreManager(session_factory, embeddings)
    logger.info(f"Creating vector store")
    
    for i in range(0, min(len(texts), max_batches * batch_size), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}: {len(batch)} documents")
        try:
            vector_store_manager.build_vector_store(batch, connection_string)
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            continue
    return session_factory, embeddings, vector_store_manager
