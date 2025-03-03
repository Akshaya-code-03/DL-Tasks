# Use official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Clone the YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# Change working directory to YOLOv5
WORKDIR /app/yolov5

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script and images folder
COPY inference.py /app/yolov5/inference.py
COPY images/ /app/yolov5/images/

# Ensure output folder exists inside the container
RUN mkdir -p /app/yolov5/output

# Run inference when the container starts
ENTRYPOINT ["python", "/app/yolov5/inference.py"]
