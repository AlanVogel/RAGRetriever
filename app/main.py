import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from vector_manager import VectorStoreManager
from rag_pipeline import RAGPipeline
from vector_db import create_vector_db, initialize_components
from logging_config import setup_logging

logger = setup_logging()
load_dotenv()

def initialize_llm(model_name: str ="mistral"):
    logger.info(f"Initializing LLM with model {model_name}")
    try:
        return OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            num_ctx=4096
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def main():
    logger.info("Creating or loading vector store")
    components = create_vector_db()
    if components:
        session_factory, embeddings, vector_store_manager = components
    else:
        logger.info("Vectore store creation failed, initializing components")
        session_factory, embeddings, _ = initialize_components()
        if not session_factory or not embeddings:
            logger.error("Failed to initialize components, exiting")
            return
        vector_store_manager = VectorStoreManager(session_factory,embeddings)

    connection_string = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@db:5432/rag_db")
    logger.info("Loading vector store")
    try:
        vector_store = vector_store_manager.load_vector_store(connection_string)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    llm = initialize_llm()
    
    logger.info("Initializing RAG pipeline")
    try:
        rag = RAGPipeline(vector_store, llm)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    question = "What are common topics in the 20 Newsgroups dataset?"
    try:
        result = rag.query(question)
        logger.info(f"Query result: {result['answer']}")
        print(f"Answer: {result["answer"]}")
        logger.info("Source documents:")
        for doc in result["source_documents"]:
            logger.info(f"{doc.page_content[:5]}...")
    except Exception as e:
        logger.error(f"Query failed: {e}")

if __name__ == "__main__":
    main()
