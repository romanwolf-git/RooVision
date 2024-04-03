# Stage 1: Builder stage
FROM python:3.10 AS builder

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY /app/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip --use-deprecated=legacy-resolver \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY /app .
COPY /yolov9 /yolov9

COPY /venv/lib/python3.10/site-packages/flask_uploads.py /usr/local/lib/python3.10/site-packages/flask_uploads.py
RUN rm requirements.txt
# Run shell command to remove the folder
RUN rm -rf /yolov9/weights
RUN rm -rf /yolov9/runs
COPY /yolov9/runs/train/18-03-2024_64Batch_300Epochs/weights/best_striped.pt /yolov9/runs/train/18-03-2024_64Batch_300Epochs/weights/best_striped.pt


# Stage 2: Runtime stage
FROM python:3.10.14-slim-bullseye
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0

# Set the working directory to /app
WORKDIR /app

# Copy the application code from the builder stage
COPY --from=builder /app .
COPY --from=builder /yolov9 /yolov9
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Expose port 8080
EXPOSE 8080

# Run wsgi.py when the container launches
ENTRYPOINT ["python", "wsgi.py"]
