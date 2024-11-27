# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    build-essential \
    g++ \
    python3-dev \
    ffmpeg \
    libmagic-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Clone the GitHub repository
ARG REPO_URL="https://github.com/naveen-s-choudhary/voice-server.git"
RUN git clone $REPO_URL .

# Debug: Verify files after cloning
RUN echo "Files after cloning:" && ls -la

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download and unzip the checkpoints
RUN curl -L -o checkpoints_v2.zip "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip" \
    && unzip checkpoints_v2.zip -d /app \
    && rm checkpoints_v2.zip

# Debug: Verify files after extracting checkpoints
RUN echo "Files after extracting checkpoints:" && ls -la

# Set the default command
CMD ["python", "index.py"]
