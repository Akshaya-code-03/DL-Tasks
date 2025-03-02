# Swin Transformer Model with AIMET Optimization

## Objective

The objective is to import the pretrained Swin-Transformer (`swin_tiny_patch4_window7_224`) model and evaluate its baseline accuracy. Then, apply Quantization, AutoQuant, and QAT on the model using AIMET and measure the accuracy results.

## Definition of Done (DoD)

- Clone the Swin-Transformer GitHub repository and install the required dependencies.
- Benchmark the baseline model on the ImageNet validation dataset (50,000 images).
- Simulate quantization of the baseline model using AIMET and evaluate accuracy.
- Apply AutoQuant and QAT with AIMET and note the accuracy.
- Compare accuracy values across baseline, quantized, AutoQuant, and QAT models.

## Setup Instructions

### Clone the Repository

```sh
git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
```

### Create a Conda Virtual Environment

```sh
conda create -n swin python=3.7 -y
conda activate swin
```

### Download Pretrained Swin-Tiny Model

```sh
mkdir checkpoints
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -P checkpoints
```

### Install Dependencies

```sh
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

## Baseline Accuracy Evaluation

Run the following command to evaluate the pretrained Swin Transformer model:

```sh
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29503 main.py \
  --eval --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
  --resume checkpoints/swin_tiny_patch4_window7_224.pth \
  --data-path /media/bmw/datasets/imagenet-1k
```

**Output:**

- Top-1 Accuracy: 81.178%
- Top-5 Accuracy: 95.520%

### Verify Standard Baseline Accuracy

Ensure that the obtained accuracy matches the standard baseline values before proceeding with further optimizations.

## Applying AIMET Optimizations

### 1. Quantization

- Simulates quantization on the model using AIMET.
- **Results:**
  - Top-1 Accuracy: 78.17%
  - Top-5 Accuracy: 94.12%

### 2. AutoQuant

- AutoQuant is applied using a subset of 1,000 images for optimization.
- **Results:**
  - Before Optimization: 0.913
  - After Optimization: 0.935

### 3. Quantization Aware Training (QAT)

- QAT is applied with AIMET using 1,000 images and batch size of 16.
- **Results:**
  - Quantized (W8A8) Accuracy: 0.904
  - After QAT: 0.912

## Comparison of Accuracy Results

| Method           | Validation Images Used | Batch Size | No. of Batches | Top-1 Accuracy | Top-5 Accuracy | Other Accuracy Metrics                    |
| ---------------- | ---------------------- | ---------- | -------------- | -------------- | -------------- | ----------------------------------------- |
| **Baseline**     | 50,000                 | 64         | 782            | 81.178%        | 95.520%        | -                                         |
| **Quantization** | 50,000                 | 64         | 782            | 78.17%         | 94.12%         | -                                         |
| **AutoQuant**    | 1,000                  | 64         | 16             | -              | -              | Before: 0.913, After: 0.935               |
| **QAT**          | 1,000                  | 16         | 63             | -              | -              | Quantized (W8A8): 0.904, After QAT: 0.912 |


