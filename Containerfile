# Start with the stable Debian Bookworm based image
# This is crucial for PyTorch binary compatibility (GLIBC)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 \
    libatomic1 \
    libopenblas0 \
    libopenblas-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fixes for "Illegal Instruction" (SIGILL) on Raspberry Pi 4
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_CORETYPE=ARMV8
# We preload both libgomp and libatomic which fixes many SIGILL cases on ARM64
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libatomic.so.1

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
