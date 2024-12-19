import streamlit as st
import requests
from PIL import Image
import io

BACKEND_URL = "http://localhost:8000"

import os
import streamlit as st
import requests

# BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')

def upload_document(uploaded_file):
    """Upload document to backend"""
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def query_document(query):
    """Send query to backend and get response"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/query", 
            data={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Query failed: {e}")
        return {"response": "Sorry, there was an error processing your query."}

def main():
    st.title("📄 Contextual Document Chat Bot")
    
    st.sidebar.header("Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", 
        type=['pdf'],
        help="Upload a PDF document to chat with"
    )
    
    st.header("Chat with Your Document")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if uploaded_file is not None:
        st.sidebar.success("Document Uploaded Successfully!")
        upload_response = upload_document(uploaded_file)
        
        if prompt := st.chat_input("Ask a question about your document"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = query_document(prompt)
                    full_response = response.get('response', 'No response found.')
                    st.markdown(full_response)
            
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.info("Please upload a PDF document in the sidebar to start chatting.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Document Chat Bot",
        page_icon="📄",
        layout="wide"
    )
    
    main()