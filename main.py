from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from retrieval_strategy import DenseVectorStrategy, KeywordSearchStrategy
from rag_generator import RagGenerator

if __name__ == "__main__":
    print("Welcome to the Local RAG Assistant! 🤖")
    
    # 1. Load our documents and database (using the classes we already built!)
    print("Loading documents and database...")
    processor = DocumentProcessor()
    chunks = processor.process_documents("data_folder")

    if not chunks:  # This means "if the list is empty"
        print("Error: No documents found in data_folder! Please add a PDF.")
        exit()
        
    db = VectorDatabase()
    vector_store = db.create_vector_store(chunks)
    
    while True:
        print("\nChoose a search method:")
        print("1. Semantic search")
        print("2. Keyword search")
        choice = input("Enter 1 or 2, or 'quit' to exit: ").strip().lower()

        if choice in {"quit", "exit"}:
            break
        if choice == "1":
            strategy = DenseVectorStrategy(vector_store)
        elif choice == "2":
            strategy = KeywordSearchStrategy(chunks)
        else:
            print("Please enter 1, 2, or 'quit'.")
            continue

        rag_generator = RagGenerator(strategy)
        question = input("Question (or 'quit' to exit): ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        answer = rag_generator.generate_response(question)
        print(f"Answer: {answer}")