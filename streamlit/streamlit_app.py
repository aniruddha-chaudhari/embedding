import streamlit as st
import requests
import os


# Get the API URL from environment variable or use default
API_URL = 'https://embedding-eight.vercel.app'

st.set_page_config(
    page_title="Vector Database Interface",
    page_icon="📚",
    layout="wide"
)

st.title("Vector Database Interface")

st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'csv'])

if uploaded_file is not None:
    with st.spinner('Uploading and processing file...'):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                st.success('File uploaded and processed successfully!')
            else:
                st.error(f'Error: {response.json().get("error", "Unknown error occurred")}')
        except Exception as e:
            st.error(f'Connection Error: {str(e)}')

