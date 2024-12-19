# Contextual Chat Bot with LLM Integration

This project implements a **Contextual Chat Bot** that processes PDF documents, extracts relevant chunks, stores embeddings in a Milvus vector database, and uses a fine-tuned LLM for querying. The bot answers questions based on the uploaded documents.

## Features:
- Upload PDF/Word documents for processing.
- Chunk and embed document text using **SentenceTransformer**.
- Store document embeddings in a **Milvus** vector database.
- Query the document using **semantic search** and generate answers via **Mistral LLM**.
- Frontend interface using **Streamlit** for uploading documents and querying the bot.

## Requirements:
- **Python 3.8+**
- **Docker** (for Milvus service)
- **Streamlit** (for frontend)
- **FastAPI** (for backend)
- **Milvus** (for vector storage)
- **PyPDFLoader** (for document loading)
- **SentenceTransformer** (for embedding generation)
- **Huggingface Transformers** (for LLM inference)

## Installation:

1. Set up Docker Compose for Milvus:
   docker-compose up -d

2. Step 1: Start the Backend (FastAPI + Milvus Integration):
uvicorn backend.src.api:app --reload --host 0.0.0.0 --port 8000

3. Step 2: Start the Frontend (Streamlit)
streamlit run frontend/streamlit.py

## MLOps Pipeline Diagram:

For a detailed view of the proposed MLOps pipeline, including model versioning, deployment, monitoring, and retraining, you can view the **draw.io** diagram here:

[**MLOps Pipeline Diagram**](https://app.diagrams.net/#G1wHFtH6ztN0gsAqE3pY9Av09VwDk18gM7)

This diagram illustrates the end-to-end workflow, from data ingestion, model training, deployment, and monitoring to model retraining, ensuring smooth integration and management of the system in production.

## Performance Evaluation

To ensure the model performs effectively in production, we use the following evaluation methods:

### 1. **Evaluation Metrics**
   - **Accuracy**: Measures the percentage of correct responses out of all predictions.
   - **Precision**: Indicates how many relevant answers were provided out of all the model's answers.
   - **Recall**: Measures how many relevant answers the model provided out of all available relevant answers.
   - **F1 Score**: Combines precision and recall into a single metric, balancing both to evaluate the model's overall performance.

These metrics are calculated by comparing the model's predictions with the ground truth, either manually or through user feedback.

### 2. **Continuous User Feedback**
   - **User Ratings**: Users provide feedback on the relevance and helpfulness of each response. This feedback is collected and analyzed to improve model performance.
   - **Error Logging**: Any failed responses or incorrect answers are logged to identify areas for improvement and retraining.

### 3. **Tracking and Monitoring**
   - **MLFlow** or similar tools can be used to track these performance metrics over time, enabling the detection of model drift or performance degradation.
   - **Reports**: Regular performance reports are generated, summarizing key metrics (accuracy, precision, recall, etc.) and highlighting trends.

### 4. **Retraining Triggers**
   - If performance drops below acceptable thresholds (e.g., accuracy, precision, or recall), retraining is triggered to update the model with new data and improve its responses.

By continually monitoring performance using these methods, we can ensure the model remains accurate, relevant, and aligned with user expectations.

