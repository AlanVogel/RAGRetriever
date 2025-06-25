from langchain_community.vectorstores import PGVector
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, Session, embeddings):
        self.Session = Session
        self.embeddings = embeddings
        self.vector_store = None

    def build_vector_store(self, texts, connection_string):
        logger.info("Building vector store")
        session = self.Session()
        try:
            self.vector_store = PGVector.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name="newsgroups_vectors",
                connection_string=connection_string
            )
            session.commit()
            logger.info("Vector store built successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to build vector store: {e}")
            raise e
        finally:
            session.close()

    def load_vector_store(self, connection_string):
        logger.info("Loading vector store")
        session = self.Session()
        try:
            self.vector_store = PGVector(
                collection_name="newsgroups_vectors",
                connection_string=connection_string,
                embedding_function=self.embeddings
            )
            session.commit()
            logger.info("Vector store loaded successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to load vector store: {e}")
            raise e
        finally:
            session.close()
        return self.vector_store
    
    def close_session(self):
        self.Session.close()
