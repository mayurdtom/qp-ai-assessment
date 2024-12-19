from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.src.document_processor import DocumentProcessor
from backend.src.vector_store import MilvusVectorStore
from backend.src.llm_service import MistralService
from fastapi import Form

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_processor = DocumentProcessor()
vector_store = MilvusVectorStore()
mistral_service = MistralService()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Process and index document
    chunks, embeddings = doc_processor.process_document(file.file)
    
    vector_store.insert_embeddings(chunks, embeddings)
    return {"status": "Document processed successfully"}

@app.post("/query")
async def query_document(query: str = Form(...)):
    # Perform semantic search
    query_embedding = doc_processor.embedder.encode([query])[0]
    search_results = vector_store.semantic_search(query_embedding)
    
    # Generate response using retrieved context
    context = " ".join([result['text'] for result in search_results])
    print(f"Context: {context}")
    response = mistral_service.generate_response(context, query)
    
    return {"query": query, "response": response}