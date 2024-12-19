import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.embedder = SentenceTransformer(embedding_model)
    
    def process_document(self, file):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        try:
            # Load and split the document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load_and_split(self.text_splitter)
            
            # Generate embeddings
            chunks = [doc.page_content for doc in documents]
            embeddings = self.embedder.encode(chunks)
        finally:
            # Clean up temporary file
            import os
            os.remove(temp_file_path)

        return chunks, embeddings
