# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first, then the rest of the dependencies
# Install CPU-only PyTorch (cached in a separate layer)
RUN pip install torch>=2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements (avoiding reinstalling torch)
RUN pip install -r requirements.txt

# Copy only necessary files
COPY serve.py .

# Expose the port for FastAPI
EXPOSE 9000

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9000"]
