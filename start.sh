#!/bin/bash
# Start the FastAPI backend on port 8000 in the background
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend on port 7860 in the foreground
python -m streamlit run streamlit_app/app.py --server.port 7860 --server.address 0.0.0.0
