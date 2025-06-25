from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from functools import partial

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, data_dir: str, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        logger.info(f"Loading documents from {self.data_dir}")
        try:
            loader = DirectoryLoader(
                self.data_dir, 
                glob="**/*.txt", 
                loader_cls=partial(TextLoader, encoding="latin-1"),
                silent_errors=True,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return []

    def split_documents(self, documents):
        logger.info("Splitting documents into chunks")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            return texts
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            return []
