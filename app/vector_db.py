import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from database import initialize_database
from document_processor import DocumentProcessor
from vector_manager import VectorStoreManager
from logging_config import setup_logging

logger = setup_logging()
load_dotenv()

def create_vector_db():
    data_dir = "./data/20_newsgroups"
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    embedding_model = "mxbai-embed-large"
    
    logger.info("Initializing database")
    Session = initialize_database(connection_string)
    
    logger.info(f"Initializing embeddings with model {embedding_model}")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    logger.info("Initializing document processor")
    processor = DocumentProcessor(data_dir)
    
    documents = processor.load_documents()
    if not documents:
        logger.warning("No documents found in the specified directory")
        return
    
    texts = processor.split_documents(documents)
    
    logger.info("Creating vector store")
    vector_store_manager = VectorStoreManager(Session, embeddings)
    vector_store_manager.build_vector_store(texts, connection_string)
