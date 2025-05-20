# Minimal Dockerfile for 0DTE Bot
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy and install runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Default entrypoint: run the 0DTE bot
ENTRYPOINT ["python3", "-m", "src.main"]
