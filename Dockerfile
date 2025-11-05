# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Update package lists and install system dependencies
# This is where we install gfortran and tesseract
RUN apt-get update && apt-get install -y \
    gfortran \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
