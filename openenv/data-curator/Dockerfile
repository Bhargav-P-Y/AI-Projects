# Use a lightweight base image to easily stay under the 8GB RAM limit
FROM python:3.11-slim

# STRICT COMPLIANCE: Hugging Face Spaces require running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire environment structure into the container
COPY . .

# Explicitly create the workspace directory to avoid read-only filesystem crashes
RUN mkdir -p /app/workspace

# Transfer ownership of the app directory to the non-root user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# STRICT COMPLIANCE: Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Start the FastAPI app, explicitly binding to 0.0.0.0 and port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
