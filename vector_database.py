class VectorDatabase:
    def __init__(self, embedding_model, vector_store_class):
        self.embedding_model = embedding_model
        self.vector_store = None  # Initialize the vector store as None

    def create_vector_store(self, documents):
        # Create embeddings for the documents
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
        # Create a vector store (e.g., FAISS) using the embeddings
        self.vector_store = FAISS.from_embeddings(embeddings, documents)