from document_processor import DocumentProcessor
from vector_database import VectorDatabase

# 1. Process the document
processor = DocumentProcessor("5008_Federalist Papers.pdf")
chunks = processor.load_and_chunk()
print(f"Created {len(chunks)} chunks.")

# 2. Create the VectorDatabase object
db = VectorDatabase()

# 3. Call the method to create the store, passing in our chunks
vector_store = db.create_vector_store(chunks)
print("Successfully created the vector database!")