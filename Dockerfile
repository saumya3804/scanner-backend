# Use a lightweight Python version
FROM python:3.9-slim

# Install the "Heavy" tools (Tesseract + Poppler)
# This command runs on the Linux cloud server
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up app folder
WORKDIR /app

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Start the server
# We bind to 0.0.0.0 so the cloud can reach it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]