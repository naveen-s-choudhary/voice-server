# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository
ARG REPO_URL="https://github.com/your-repository.git"
RUN git clone $REPO_URL .

# Change to the `voice` directory
WORKDIR /app/voice

# Download and extract checkpoints_v2.zip
RUN curl -L -o checkpoints_v2.zip "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip" \
    && unzip checkpoints_v2.zip -d /app/voice \
    && rm checkpoints_v2.zip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command
CMD ["python", "index.py"]
