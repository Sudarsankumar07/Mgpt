# Use Python 3.10 base image with Debian Bookworm, which has SQLite >= 3.40.1 (meets ChromaDB's requirement)
FROM python:3.10-bookworm

# Install system dependencies, including libsqlite3-dev for ChromaDB
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify SQLite version (should be >= 3.35.0)
RUN sqlite3 --version

# Set working directory
WORKDIR /app

# Copy your code and requirements
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]