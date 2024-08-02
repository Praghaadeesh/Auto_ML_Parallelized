# Use an official Python runtime as the base image
FROM python:3.11.5

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY requirements.txt ./
COPY app.py ./
COPY templates/index.html ./
COPY templates/result.html ./
COPY static/background_image.png ./
COPY static/result_graph.png ./

# Install any project-specific dependencies
RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000

# Specify the command to run your project
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]