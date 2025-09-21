# Use Python 3.11 base image with Debian Bookworm (SQLite >= 3.40.1)
FROM python:3.11-bookworm

# Install system dependencies, including libsqlite3-dev for ChromaDB
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directory for ChromaDB PersistentClient
RUN mkdir -p ./vector_db

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]