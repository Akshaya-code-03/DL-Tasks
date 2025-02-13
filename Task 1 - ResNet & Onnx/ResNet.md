# ResNet

### What is ResNet?

- ResNet ( Residual Network ) - is a type of deep learning architecture designed to solve the **vanishing gradient problem** in deep neural networks
- Using ResNets we can train extremely deep neural networks with 150+ layers

---

### Gradient

A measure of how much a neural network's weights change in relation to how much the error changes

### Vanishing Gradient Problem

When number of layers is high - In deep neural networks, when we backpropagate gradients (updates for weights), they **become smaller and smaller** as they move backward through many layers. This causes:

✅ **Early layers to learn very slowly**

✅ **Training to become ineffective**

- During backpropagation, if gradients **become very small**, the **weight updates shrink** drastically:
    
    W=W−η⋅∇L
    
    - W = Weight
    - η = Learning rate
    - ∇L = Gradient of the loss function
- Since ∇L is **almost zero**, the weight update is **too small**.
- As a result, **weights remain nearly unchanged** over multiple epochs.
- This **prevents early layers from learning**, while later layers still receive significant updates.

![image.png](050bf616-3d61-48c4-ae61-678072e6bd4e.png)

---

### How does ResNet handle Vanishing Gradient Problem ?

**Residual Connections (Skip Connections):**

1. Residual connections are shortcuts that **bypass one or more layers** and directly connect the input to the output of a deeper layer instead of direct connection ( Output = Weight * input + bias ). 
2. It groups layers of networks into blocks and for each block make the data go through and around.
3. This helps gradients to flow easily during backpropagation, solving the vanishing gradient issue.

        *`Output - Input = Residual` ⇒  `Input + Residual = Output`*

- Instead of making the network learn output **from scratch**, we help it by assuming that the input x is already a good starting point.
- Now, the network only needs to learn the small difference F(x) (the "residual").

**Example:**

- Suppose x is an image of a cat.
- The next layer should detect "edges of the cat."
- Instead of learning the full transformation, the layer only learns **"what is missing"** (residual), which is easier!

![image.png](image.png)

---

### **Why Learning Residuals Helps?**

- Small changes are easier to learn than the full transformation.
- Ensures **stable gradient flow** because we always add x, keeping information from earlier layers.
- Reduces the risk of vanishing gradients.

---

### **When Do We Skip? (Criteria for Skipping Layers)**

### ✅ **Condition 1: Input and Output Shapes Must Match**

- If the input to a block and the output of the transformation have **the same dimensions**, we **directly add** them.
- If they don’t match, we **use a 1×1 convolution** to adjust dimensions before adding incase of more number of channels requirement and use a **stride of 2** when the height and width needs to be reduced to hald of its value.
- **Why 1×1 Convolution?
      →** Doesn’t change spatial dimensions (height & width remain the same).
      → Only modifies depth (number of channels) to match the output shape.
    
    

### ✅ **Condition 2: Every Residual Block Has a Skip Connection**

- ResNet is built using **residual blocks**:
    - Each block contains **two or three layers** (Conv + BatchNorm + ReLU).
    - The **skip connection jumps over the entire block**, not individual layers inside it.
- This ensures deeper networks still **benefit from depth** but avoid vanishing gradients.

![image.png](image%201.png)

✅ **Condition 3: Deeper Networks Use More Skip Connections**

- **Shallower ResNets (ResNet-18, 34)** → Use **basic residual blocks** (2 layers per block).
- **Deeper ResNets (ResNet-50, 101, 152)** → Use **bottleneck residual blocks** (3 layers per block).
- This makes deeper models train efficiently **without extra computation overhead**.

---

https://youtu.be/Q1JCrG1bJ-A?si=tlX35Ufe44HMGg1H

![image.png](image%202.png)

---

### What is ResNet18?

**ResNet-18** is a deep convolutional neural network (CNN) with **18 layers**, designed for image classification and feature extraction.

---

### **Breakdown of Layers in ResNet-18**

1. **Initial convolution + max pooling** (1 layer)
2. **8 Residual Blocks** → Each block has **2 convolutional layers**
    - 2×8=16 convolutional layers
3. **Final fully connected (FC) layer** (1 layer)

📌 **Total weighted layers** → **1 (initial) + 16 (residual) + 1 (FC) = 18**