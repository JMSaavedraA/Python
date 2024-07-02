# Use the official Python base image
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --upgrade numpy matplotlib scikit-image Pillow tqdm jupyter jupyterlab notebook
