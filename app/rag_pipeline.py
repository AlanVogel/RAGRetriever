from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from logging_config import setup_logging

logger = setup_logging()
class RAGPipeline:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.qa_chain = None
        self._setup_qa_chain()

    def _setup_qa_chain(self):
        logger.info("Setting up QA chain")
        try:
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
        except Exception as e:
            logger.error(f"Failed to set up QA chain: {e}")
            self.qa_chain = None

    def query(self, question: str):
        logger.info(f"Processing query: {question}")
        if not self.qa_chain:
            logger.error("QA chain not initialized")
            raise ValueError("QA chain not initialized")
        try:
            result = self.qa_chain.invoke({"query": question})
            logger.info("Query processed successfully")
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    