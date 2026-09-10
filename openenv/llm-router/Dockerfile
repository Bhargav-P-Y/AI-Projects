# STRICT COMPLIANCE: Match local development environment
FROM python:3.12-slim

# STRICT COMPLIANCE: Hugging Face Spaces require running as a non-root user (UID 1000)
RUN groupadd -r user && useradd -m -r -g user -u 1000 user

# Set the working directory
WORKDIR /app

# Install uv (The modern, fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy the dependency files first (for Docker caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv system-wide
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Transfer ownership of the app directory to the non-root user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# STRICT COMPLIANCE: Expose the standard Hugging Face Space port
EXPOSE 7860

# Command to run the FastAPI server (explicit host binding required)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
