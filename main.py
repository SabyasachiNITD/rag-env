from document_processor import DocumentProcessor

# 1. Create an object of our class
processor = DocumentProcessor("5008_Federalist Papers.pdf")

# 2. Call the method to process the document
chunks = processor.load_and_chunk()

# 3. Verify it worked
print(f"Success! We created {len(chunks)} chunks.")