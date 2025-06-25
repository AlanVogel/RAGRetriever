import os
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from database import initialize_database
from document_processor import DocumentProcessor
from vector_manager import VectorStoreManager
from logging_config import setup_logging

logger = setup_logging()

def create_vector_db(batch_size=100, retry_delay=2, max_batches=300):
    load_dotenv()
    data_dir = "./data/20_newsgroups"
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    embedding_model = "all-minilm"
    
    logger.info("Initializing database")
    Session = initialize_database(connection_string)
    
    logger.info(f"Initializing embeddings with model {embedding_model}")
    embeddings = OllamaEmbeddings(
        model = embedding_model, 
        base_url = os.getenv("OLLAMA_URL", "http://ollama:11434"),
        num_ctx=2048,
        temperature=None,
        mirostat=None,
        mirostat_eta=None,
        mirostat_tau=None,
        tfs_z=None
    )
    
    logger.info("Initializing document processor")
    processor = DocumentProcessor(data_dir)
    
    documents = processor.load_documents()
    if not documents:
        logger.warning("No documents found in the specified directory")
        return
    logger.info(f"Total documents loaded: {len(documents)}")

    texts = processor.split_documents(documents)
    logger.info("Creating vector store")
    if len(texts) > 10000:
        logger.warning(f"High chunk count ({len(texts)}), consider increasing chunk_size")

    vector_store_manager = VectorStoreManager(Session, embeddings)
    
    for i in range(0, min(len(texts), max_batches * batch_size), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}: {len(batch)} documents")
        try:
            vector_store_manager.build_vector_store(batch, connection_string)
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            time.sleep(retry_delay)
    vector_store_manager.close_session()
