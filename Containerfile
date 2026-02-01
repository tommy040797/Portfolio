# Start with a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Basic tools and Pillow/Torch deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 \
    libopenblas0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (on ARM64, the standard index works best for Torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose the port FastAPI runs on
EXPOSE 80

# Set the working directory to where main.py is
WORKDIR /app/backend

# Create uploads directory explicitly
RUN mkdir -p temp_uploads

# Command to run the application
# Using python directly to run the __main__ block in main.py
CMD ["python", "main.py"]
