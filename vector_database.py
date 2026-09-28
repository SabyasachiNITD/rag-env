from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class VectorDatabase:
    def __init__(self, model_name="nomic-embed-text", persist_directory="./chroma_db"):
        self.embedding_model = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434"
        )
        self.persist_directory = persist_directory

    def create_vector_store(self, chunks):
        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            self.vector_store.add_documents(chunks[start:start + batch_size])
        return self.vector_store