from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def load_and_chunk(self, file_path=None):
        path = file_path or self.file_path
        if path is None:
            raise ValueError("Provide a PDF path to load_and_chunk().")

        loader = PyPDFLoader(str(path))
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def process_documents(self, directory_path):
        directory = Path(directory_path)
        chunks = []
        for pdf_path in sorted(directory.glob("*.pdf")):
            chunks.extend(self.load_and_chunk(pdf_path))
        return chunks