# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Upgrade pip to the latest version
RUN python -m pip install --no-cache-dir --upgrade pip

# Install the Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Remove the temporary requirements.txt file
RUN rm requirements.txt

# Copy the entire source code to the container
COPY . .

# Expose port 8022
EXPOSE 8022

# Expose Port 8021
EXPOSE 8021

# Copy the entrypoint script to the container
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Set the entrypoint to run the scripts
ENTRYPOINT ["./entrypoint.sh"]

# Run script1 on port 8022
# CMD ["python", "src/main.py"] #docker change test