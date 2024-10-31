FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["python", "serve_model.py"]
