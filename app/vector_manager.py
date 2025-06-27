from langchain_community.vectorstores import PGVector
from logging_config import setup_logging

logger = setup_logging()

class VectorStoreManager:
    def __init__(self, session_factory, embeddings, collection_name="newsgroups_vectors"):
        self.session_factory = session_factory
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.vector_store = None

    def build_vector_store(self, documents, connection_string):
        logger.info("Building vector store")
        try:
            with self.session_factory() as session:
                self.vector_store = PGVector.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_string=connection_string,
                    use_jsonb=True
                )
                session.commit()
            logger.info("Vector store built successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to build vector store: {e}")
            raise

    def load_vector_store(self, connection_string):
        logger.info("Loading vector store")
        try:
            with self.session_factory() as session:
                self.vector_store = PGVector(
                    collection_name=self.collection_name,
                    connection_string=connection_string,
                    embedding_function=self.embeddings,
                    use_jsonb=True
                )
            logger.info("Vector store loaded successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
