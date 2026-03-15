#!/bin/bash
# Install lightweight dependencies for backend only
pip install -r requirements-backend.txt

# Start Backend API (FastAPI)
uvicorn chatbot_api.server:app --host 0.0.0.0 --port ${PORT:-8000}
