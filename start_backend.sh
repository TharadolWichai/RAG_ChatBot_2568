#!/bin/bash
# Start Backend API (FastAPI)
# Railway auto-installs requirements.txt
uvicorn chatbot_api.server:app --host 0.0.0.0 --port ${PORT:-8000}
