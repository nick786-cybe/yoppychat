# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cookies file into the container
COPY cookies.txt .
# Copy the rest of the application's code into the container
COPY . .

# Expose the port the web app runs on
EXPOSE 5000

# We will remove the CMD from here and define it in docker-compose.yml
# This makes our Docker image more flexible.
