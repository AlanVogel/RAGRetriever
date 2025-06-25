import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from database import initialize_database
from document_processor import DocumentProcessor
from vector_manager import VectorStoreManager
from rag_pipeline import RAGPipeline
from vector_db import create_vector_db
from logging_config import setup_logging

logger = setup_logging()

def initialize_llm(model_name: str ="mistral"):
    logger.info(f"Initializing LLM with model {model_name}")
    try:
        return OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            num_ctx=2048
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def main():
    load_dotenv()
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    
    logger.info("Initializing dependencies")
    Session = initialize_database(connection_string)
    try:
        embeddings = OllamaEmbeddings(
            model="all-minilm",
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            num_ctx=2048,
            temperature=None,
            mirostat=None,
            mirostat_eta=None,
            mirostat_tau=None,
            tfs_z=None
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")

    llm = initialize_llm()
    
    logger.info("Loading vector store")
    vector_store_manager = VectorStoreManager(Session, embeddings)
    vector_store = vector_store_manager.load_vector_store(connection_string)
    
    logger.info("Initializing RAG pipeline")
    rag = RAGPipeline(vector_store, llm)
    
    query = "What are common topics in the 20 Newsgroups dataset?"
    result = rag.query(query)
    
    logger.info(f"Query result: {result['answer']}")
    logger.info("Source documents:")
    for doc in result["source_documents"]:
        logger.info(f"{doc.page_content[:100]}...")
    
    vector_store_manager.close_session()

if __name__ == "__main__":
    create_vector_db()
    main()
