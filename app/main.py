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

def initialize_llm(model_name: str ="llama4"):
    logger.info(f"Initializing LLM with model {model_name}")
    return OllamaLLM(model=model_name)

def main():
    load_dotenv()
    data_dir = "./data/20_newsgroups"
    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    
    logger.info("Initializing dependencies")
    Session = initialize_database(connection_string)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    llm = initialize_llm()
    
    logger.info("Initializing document processor")
    processor = DocumentProcessor(data_dir)
    
    documents = processor.load_documents()
    if documents:
        texts = processor.split_documents(documents)
        vector_store_manager = VectorStoreManager(Session, embeddings)
        vector_store_manager.build_vector_store(texts, connection_string)
    
    logger.info("Initializing RAG pipeline")
    rag = RAGPipeline(Session, llm, connection_string)
    
    query = "What are common topics in the 20 Newsgroups dataset?"
    result = rag.query(query)
    
    logger.info(f"Query result: {result['answer']}")
    logger.info("Source documents:")
    for doc in result["source_documents"]:
        logger.info(f"{doc.page_content[:100]}...")

if __name__ == "__main__":
    create_vector_db()
    main()
