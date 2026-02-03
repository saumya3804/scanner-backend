# Use a specific, stable version of Python/Linux (Bullseye)
FROM python:3.9-slim-bullseye

# Install System Dependencies
# We add --fix-missing and extra graphic libraries (libsm6, libxext6) to prevent crashes
RUN apt-get update --fix-missing && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up app folder
WORKDIR /app

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Start the server

CMD ["uvicorn", "opencv:app", "--host", "0.0.0.0", "--port", "8000"]
