# Use lightweight image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set FastAPI to run
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

