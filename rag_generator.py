from langchain_ollama import ChatOllama

class RagGenerator:
    def __init__(self, retrieval_strategy):
        self.chat_model = ChatOllama(
            model="llama3.2:3b",
            base_url="http://localhost:11434"
        )
        self.retrieval_strategy = retrieval_strategy

    def generate_response(self, query: str):
        # 1. Retrieve relevant chunks
        relevant_chunks = self.retrieval_strategy.retrieve(query)

        # 2. Extract the text from the chunks
        documents = []
        for chunk in relevant_chunks:
            documents.append(chunk.page_content)
        documents_str = " ".join(documents)
        
        # 3. Build the prompt
        prompt = f"Answer the following question based on the provided documents: {query}\n\nDocuments: {documents_str}"
        
        # 4. Generate and return the final answer
        ai_message = self.chat_model.invoke(prompt)
        return ai_message.content