# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir flask torch torchvision pillow

# Expose port 8888 to the outside world
EXPOSE 8888

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8888 

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8888"]
