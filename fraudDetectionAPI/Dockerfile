# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt separately to leverage Docker caching
COPY requirements.txt requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with timeout to avoid network issues
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the correct application file
CMD ["python", "serve_model.py"]
