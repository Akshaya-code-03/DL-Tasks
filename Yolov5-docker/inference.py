import torch
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define input and output directories
input_folder = "images"
output_folder = "output"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all image files from the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Run inference on each image
for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    
    # Perform object detection
    results = model(img_path)
    
    # Save results to output folder
    output_path = os.path.join(output_folder, img_file)
    results.save(save_dir=output_path)

    print(f"Inference done for {img_file}. Results saved to {output_path}")
