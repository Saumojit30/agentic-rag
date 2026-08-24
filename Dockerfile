# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for compiling certain packages if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose ports (FastAPI on 8000, Streamlit on 7860)
EXPOSE 8000
EXPOSE 7860

# Make start script executable
RUN chmod +x start.sh

# Run startup script
CMD ["./start.sh"]
