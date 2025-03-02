# YOLOv5 Object Detection with Docker

## Overview
This project performs object detection on sample images using the YOLOv5 model and exports it as a Docker container.

## Project Structure
```
Yolov5_docker/
│── images/           # Folder where you will drop input images
│── output/           # Folder where the inference results will be saved
│── inference.py      # Python script to perform inference
│── Dockerfile        # Docker configuration
```

## Requirements
- Docker installed on your system
- Internet connection to download dependencies

## Steps to Run the Dockerized YOLOv5 Inference

### 1. Build the Docker Image
Run the following command to build the Docker image:
```sh
docker build -t Yolov5_docker .
```
This will:
- Install Python dependencies
- Download the YOLOv5 model
- Set up the inference environment

### 2. Run the Docker Container
Run the following command to perform inference on images:
```sh
docker run --rm -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output Yolov5_docker
```

### 3. Verify Docker Image Creation
To list available Docker images:
```sh
docker images
```

### 4. Save Docker Image (Optional)
To export the Docker image as a `.tar` file:
```sh
docker save -o Yolov5_docker.tar Yolov5_docker
```

## Inference Script (inference.py)
This script:
1. Loads the YOLOv5 model
2. Fetches images from the `images/` folder
3. Runs inference on each image
4. Saves results to the `output/` folder

## Dockerfile
This Dockerfile:
- Uses Python as the base image
- Installs dependencies
- Copies the inference script and model
- Runs inference when the container starts

## Credits
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)

