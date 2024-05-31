# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of your app's code into the container at /app
COPY . .

# Expose port 8501 to the outside world
EXPOSE 1234

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py",  "--server.port", "1234"]
