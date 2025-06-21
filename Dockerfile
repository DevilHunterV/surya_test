# Use latest Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install pip & tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy rest of the app
COPY . .

# Run your script
CMD ["python", "forecast_runner.py"]
