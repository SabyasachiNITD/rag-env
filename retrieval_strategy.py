from abc import ABC, abstractmethod
from langchain_community.retrievers import BM25Retriever

class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str):
        pass

class DenseVectorStrategy(RetrievalStrategy):
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def retrieve(self, query: str):
        # We need to search the vector_store here
        results = self.vector_store.similarity_search(query)
        return results

class KeywordSearchStrategy(RetrievalStrategy):
    def __init__(self, chunks):
        # We build the keyword index from the raw text chunks
        self.retriever = BM25Retriever.from_documents(chunks)
        
    def retrieve(self, query: str):
        # We need to search the vector_store here
        return self.retriever.invoke(query)