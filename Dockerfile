# Use Python 3.11 base image with Debian Bookworm (SQLite >= 3.40.1)
FROM python:3.11-bookworm

# Install system dependencies first (cached layer)
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements.txt first (for caching pip install)
COPY requirements.txt .

# Install Python dependencies (cached if requirements.txt unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code (changes here won't invalidate dependency layers)
COPY . .

# Create directory for ChromaDB PersistentClient
RUN mkdir -p ./vector_db

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]