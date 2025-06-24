from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_manager import VectorStoreManager
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, Session, llm, connection_string: str):
        self.vector_store_manager = VectorStoreManager(Session, None)
        self.vector_store = self.vector_store_manager.load_vector_store(connection_string)
        self.llm = llm
        self.qa_chain = self._setup_qa_chain()

    def _setup_qa_chain(self):
        logger.info("Setting up QA chain")
        prompt_template = """Use the following context to answer the question:
        {context}
        
        Question: {question}
        Answer: """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("QA chain setup complete")

    def query(self, question: str):
        logger.info(f"Processing query: {question}")
        if not self.qa_chain:
            logger.error("QA chain not initialized")
            raise ValueError("QA chain not initialized")
        result = self.qa_chain({"query": question})
        logger.info("Query processed successfully")
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    