#!/bin/bash
# Install full dependencies for frontend (includes streamlit)
pip install -r requirements-full.txt

# Start Chatbot UI (Streamlit)
streamlit run automated_data_ingestion/dashboard/chatbot_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
