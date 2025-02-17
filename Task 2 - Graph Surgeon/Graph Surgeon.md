# Graph Surgeon

### TERMINOLOGIES:

- **Inference:** In AI, **inference** means using a trained model to **make predictions** on new data.
- **Runtime Library:** It provides **pre-built functions and optimizations** so the AI model can run smoothly on hardware without worrying about the low-level details.
- **GPU: GPU** can handle **thousands of small tasks** at once.
- **Layer Fusion: Layer fusion** combines multiple layes into **one optimized layer**, making inference **faster**.
- **Memory Footprint:** Memory footprint is **the amount of memory a program uses** while running.
- **Precision Calibration:** Precision calibration means **adjusting the accuracy of numbers** in an AI model to make it **faster** and **use less memory**.
    
             AI models use **floating-point numbers** like:
    
    - **FP32 (32-bit floating-point)** – Very precise but **slow**. Eg: 2.743
    - **FP16 (16-bit floating-point)** – Less precise but **faster**. Eg: 2.74
    - **INT8 (8-bit integer)** – Even less precise but **super fast** and uses less memory. Eg: 2.7
- **Kernel Auto-tune:** A **kernel** in AI is a small piece of code that runs on a **GPU** to perform **calculations. Auto-tuning** means TensorRT **automatically chooses the best kernel** for your GPU to get **maximum speed and efficiency**.
- **Throughput:** The amount of product or service that a company can produce and deliver within a specified period of time

---

### What is TensorRT?

- TensorRT ( Tensor Runtime ) is a **high-performance deep learning inference optimizer and runtime library** developed by NVIDIA.
- It is designed to accelerate deep learning models for **inference on NVIDIA GPUs**.

---

### Key Features:

1. **Model Optimization** – Performs layer fusion, precision calibration (FP32, FP16, INT8), and kernel auto-tuning for better efficiency.
2. **Low Latency & High Throughput** – Optimized for real-time inference applications like AI in self-driving cars, healthcare, and NLP.
3. **Supports Multiple Frameworks** – Converts models from TensorFlow, PyTorch, ONNX, and more into optimized TensorRT engines.
4. **Efficient Memory Management** – Reduces memory footprint and optimizes execution for better GPU utilization.

---

### Graph Surgeon:

- **Graph-Surgeon** is a tool in TensorRT that helps modify and optimize **computational graphs** before running them on a GPU.
- It helps remove unnecessary layers, fuse operations, and optimize the model for deployment.

---

### ONNX GRAPH:

An **ONNX graph** represents a deep learning model as a **Directed Acyclic Graph (DAG)**, where:

- **Nodes** = Operations (e.g., Conv, ReLU, BatchNorm)
- **Edges** = Data flow between layers

---

### **Why Use Graph-Surgeon?**

When a deep learning model is trained, it contains extra layers that are **only useful for training** but **not needed for inference**.

Graph-Surgeon helps by:

✅ Removing **training-only layers** (e.g., Dropout, Loss layers)

✅ **Fusing operations** (e.g., combining BatchNorm + Convolution)

✅ **Replacing unsupported layers** (e.g., converting complex ONNX operations into TensorRT-supported ones)

✅ **Pruning unused nodes** to make the graph efficient

---

### **Main Components of ONNX GraphSurgeon**

ONNX GraphSurgeon has three major parts:

1️⃣ **Importers** – Load an ONNX model into GraphSurgeon for modification.

2️⃣ **Intermediate Representation (IR)** – The core data structure where all changes are made.

3️⃣ **Exporters** – Save the modified model back into ONNX format.

---

### **1. Importers – Loading an ONNX Model**

Importers help load a model into the ONNX GraphSurgeon **Intermediate Representation (IR)**.

🔹 **Example: Importing an ONNX model**

```python
import onnx
import graphsurgeon as gs

# Load an ONNX model
onnx_model = onnx.load("model.onnx")

# Convert ONNX model into GraphSurgeon format
graph = gs.import_onnx(onnx_model)
```

---

### **2. Intermediate Representation (IR) – The Core of GraphSurgeon**

The **Intermediate Representation (IR)** is where the model is stored as a **Graph** containing:

- **Tensors** – Store data
- **Nodes** – Perform operations
- **Graph** – The overall model

**(i) Tensors – Store Data**

Tensors represent the **data flowing through the model**. There are two types:

| Tensor Type | Description |
| --- | --- |
| **Constant** | Fixed values known before inference (like weights & biases). |
| **Variable** | Unknown until inference-time (like input images, dynamic values). |

**Constant:**

```python
print(tensor)
Constant (name_of_the_tensor)
[0.85, 1.15, 0.91, ...]  # Fixed weights
```

**Variable:**

```python
print(tensor)
Variable (name_of_the_tensor): (shape=[1, 3, 224, 224], dtype=float32)
"""
1 → Batch size
3 → Number of channels (like RGB)
224 x 224 → Height and width of the image
dtype=float32 → Data type is 32-bit floating point
"""
```

**ii) Nodes – Perform Operations**

Nodes represent **mathematical operations** like Convolution, ReLU, etc.

They **take input tensors, apply a function, and produce output tensors**.

🔹 **Example ReLU Node in ResNet-50**

```
print(node)
(Relu)
    Inputs: [Tensor (tensor_name)]
    Outputs: [Tensor (tensor_name)]
```

**(iii) Graph – The Overall Model**

A Graph contains:

- **Nodes (Operations)**
- **Tensors (Data)**
- **Input/Output Connections**

🔹 **Functions in GraphSurgeon to Modify Graphs**

| Function | Description |
| --- | --- |
| **cleanup()** | Removes unused nodes/tensors
graph.cleanup(remove_unused_nodes=True) |
| **toposort()** | Sorts nodes in topological order |
| **tensors()** | Returns all tensors in the graph |

---

### **3. Exporters – Saving the Modified Model**

After modifying the model, we save it back to ONNX format.

**Example: Exporting an Optimized ONNX Model**

```python
onnx.save(gs.export_onnx(graph), "optimized_model.onnx")
```

---

### Working with Graph Surgeon - Reference

https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples

---

### **What is GlobalLpPool?**

**GlobalLpPool** is a type of **pooling operation** used in neural networks. It reduces the **spatial dimensions** (height and width) of an input tensor to **1×1**, keeping only the depth (channels) unchanged.

---

### **How It Works?**

- It applies **Lp-norm pooling** across the entire spatial area.

$$
Y = ( ∑ ∣X∣ ^ p ) ^ (1/p)

$$

- p is the **norm parameter** (e.g., **p=2** for L2 pooling).
    - **If p=1 (L1 Pooling) :** Takes the **sum of absolute values** across the spatial area. For sparse feature learning
    - **If p=2 (L2 Pooling):** Computes the **root mean square (RMS)** of the values. For energy-preserving representations
    - **If p=∞ (Max Pooling):** Equivalent to **Global Max Pooling** (takes the maximum value). For maximum activation selection
- X is the input.
- Y is the output.

---

### **Why Use GlobalLpPool?**

1. **Feature Extraction** – Helps extract the most important features from an image.
2. **Dimensionality Reduction** – Reduces computation by decreasing spatial size.
3. **Improves Model Generalization** – Keeps key information while ignoring unimportant details.

---

### Difference between torch.onnx and onnx

| Feature | `torch.onnx` | `onnx` |
| --- | --- | --- |
| **Purpose** | Converts PyTorch models to ONNX format | Loads, checks, and manipulates ONNX models |
| **Usage** | Used during model export | Used after model export for verification, modification, and inference |
| **Key Functions** | `torch.onnx.export()` | `onnx.load()`, `onnx.checker.check_model()` |
| **Framework** | Works with PyTorch models | Works with ONNX models |
| **Dependency** | Requires PyTorch (`torchvision` for pretrained models) | Does not require PyTorch |

---

### **Folding in ONNX Graph**

Folding in an ONNX graph refers to **simplifying or optimizing** the computational graph by **removing redundant operations** and **precomputing constant expressions** to improve inference efficiency.

### **Types of Folding in ONNX**

1. **Constant Folding**
    - **Precomputes operations** that involve only constant values.
    - Example: If an ONNX model has `3 + 5`, it replaces it with `8` at graph optimization time.
    - **Benefits**: Reduces unnecessary computation at runtime.
    
    **LIMITATION:**
    
    - ONNX GraphSurgeon's built-in constant folding doesn’t support rotation of nodes
    - **Rotation of nodes** means **changing the order of operations** while preserving output.
    - Example:
        - **Expression That Will Be Folded:**
        
        $$
        x+(c0​+(c1​+c2​))
        $$
        
        - Here, **c0,c1,c2 are constants**, and they are **nested inside parentheses** in a way that allows **constant folding first** before adding x.
        - The computation proceeds as:
            1. **Fold constants first:** c1+c2=C′
            2. **Then fold the result with another constant:** c0+C′=C′′
            3. **Finally, add the variable:** x+C′′
            
            - **Expression That Will NOT Be Folded:**
            
            $$
            ((x+c0​)+c1​)+c2​
            $$
            
2. **BatchNorm Folding**
    - **Merges Batch Normalization into Convolution layers** to reduce computations.
    - Example: A BatchNorm layer after Conv can be mathematically merged into Conv weights.
    - **Benefits**: Reduces memory access and speeds up inference.
3. **Transpose Folding**
    - **Removes unnecessary transpose operations** by simplifying input/output tensor orders.
    - Example: If two consecutive `Transpose` operations cancel out, they are removed.
    - **Benefits**: Optimizes memory layout and computation.
4. **Identity Folding**
    - **Removes Identity nodes** that don’t modify the tensor.
    - Example: `y = Identity(x)` is removed if `y` is always the same as `x`.
    - **Benefits**: Reduces computation graph complexity.
5. **Reshape Folding**
    - **Merges unnecessary reshape operations** into one.
    - Example: `Reshape -> Reshape -> Reshape` can be collapsed into a single `Reshape`.
    - **Benefits**: Avoids redundant operations.

---

### Types of ONNX Nodes:

### **1️⃣ Mathematical Nodes (Basic Operations)**

- **Add** – Performs element-wise addition of two tensors.
- **Sub** – Performs element-wise subtraction of two tensors.
- **Mul** – Performs element-wise multiplication of two tensors.
- **Div** – Performs element-wise division of two tensors.
- **Exp** – Applies the exponential function to each element.
- **Log** – Applies the natural logarithm to each element.
- **Pow** – Raises each element to a specified power.
- **Abs** – Computes the absolute value of each element.
- **Neg** – Negates each element in the tensor.
- **Sum** – Computes the sum of multiple input tensors.

### **2️⃣ Activation Nodes**

- **Relu** – Replaces negative values with zero (Rectified Linear Unit).
- **LeakyRelu** – Allows a small negative slope (α) for negative values instead of zero. α is fixed.
- **PRelu** – A learnable version of LeakyReLU with trainable slope. α is variable.
- **Sigmoid** – Maps values to a range between 0 and 1, used in binary classification.
- **Tanh** – Maps values to a range between -1 and 1, commonly used in RNNs.
- **Softmax** – Converts logits into probability distributions for classification.
- **ThresholdedRelu** – Similar to ReLU but only activates values above a threshold.
- **Elu** – Exponential Linear Unit, smooth variant of ReLU.
- **HardSigmoid** – An approximation of the Sigmoid function for efficiency.
- **HardSwish** – An efficient version of the Swish ( A self-gated activation function that **multiplies input by its sigmoid value -** used exponentiation function ) activation function - used **piecewise linear** function (max(1, x + 3) / 6)

### **3️⃣ Convolution and Pooling Nodes**

- **Conv** – Applies a convolution operation (feature extraction).
- **ConvTranspose** – Performs transposed convolution (upsampling).
- **MaxPool** – Reduces spatial dimensions by taking the maximum value in a window.
- **AveragePool** – Reduces spatial dimensions by averaging values in a window.
- **GlobalMaxPool** – Applies max pooling across the entire spatial dimensions.
- **GlobalAveragePool** – Applies average pooling across the entire spatial dimensions.
- **LpPool** – Performs pooling using Lp-norm instead of max/average.

### **4️⃣ Normalization Nodes**

- **BatchNormalization** – Normalizes activations across a mini-batch for stable training.
- **LayerNormalization** – Normalizes across features within a layer.
- **InstanceNormalization** – Normalizes per image/sample instead of across a batch.
- **LpNormalization** – Normalizes tensor values using Lp-norm.

### **5️⃣ Tensor Manipulation Nodes**

- **Reshape** – Changes the shape of a tensor without changing data.
- **Transpose** – Rearranges dimensions of a tensor.
- **Concat** – Joins multiple tensors along a specific axis.
- **Split** – Splits a tensor into multiple parts along an axis.
- **Slice** – Extracts a portion of a tensor using indices.
- **Squeeze** – Removes dimensions of size 1 (e.g., from [1,3,224,224] to [3,224,224]).
- **Unsqueeze** – Adds dimensions of size 1 to a tensor.
- **Flatten** – Converts a multi-dimensional tensor into a 1D tensor.
- **Expand** – Expands a tensor to match a target shape via broadcasting.

### **6️⃣ Reduction Nodes**

- **ReduceSum** – Computes the sum of elements along an axis.
- **ReduceMean** – Computes the mean of elements along an axis.
- **ReduceMax** – Finds the maximum value along an axis.
- **ReduceMin** – Finds the minimum value along an axis.
- **ReduceProd** – Computes the product of elements along an axis.

### **7️⃣ Element-wise Comparison Nodes**

- **Equal** – Checks if elements of two tensors are equal.
- **Greater** – Checks if elements of one tensor are greater than another.
- **Less** – Checks if elements of one tensor are smaller than another.
- **GreaterOrEqual** – Checks if elements are greater than or equal to another tensor.
- **LessOrEqual** – Checks if elements are less than or equal to another tensor.

### **8️⃣ Logical Nodes**

- **And** – Performs element-wise logical AND operation.
- **Or** – Performs element-wise logical OR operation.
- **Xor** – Performs element-wise logical XOR operation.
- **Not** – Performs element-wise logical NOT operation.

### **9️⃣ Control Flow Nodes**

- **If** – Executes one of two branches based on a condition.
- **Loop** – Executes a loop a given number of times.
- **Scan** – Iterates over a sequence of tensors.

### **🔟 Data Handling Nodes**

- **Identity** – Passes the input tensor unchanged (useful in debugging).
- **Constant** – Represents a constant tensor in the model.
- **Gather** – Selects elements from a tensor based on indices.
- **Scatter** – Writes values into a tensor at specified indices.
- **Shape** – Retrieves the shape of a tensor.
- **Size** – Gets the total number of elements in a tensor.

### **11. Statistical Nodes**

- **ArgMax** – Returns the index of the maximum value along an axis.
- **ArgMin** – Returns the index of the minimum value along an axis.
- **TopK** – Returns the top K largest values and their indices.

### **12. Image Processing Nodes**

- **Resize** – Rescales a tensor (image) to a new size.
- **Crop** – Extracts a cropped region from a tensor.

**NOTE:**

GraphSurgeon allows defining **custom nodes**, so you can name a node anything.

---

### **What is `Graph.layer()`?**

The `Graph.layer()` function in **GraphSurgeon (gs)** provides a **simplified way to add nodes to an ONNX graph**.

Instead of manually defining each node, input, and output tensor separately, you can use `Graph.layer()` to:

- Create a new node (e.g., `Add`, `Gemm`, `Relu`, etc.).
- Define its input and output tensors.
- Automatically insert the node into the graph.

This makes model construction **easier and more readable** compared to manually defining and linking every node.

**Example:**

Instead of writing a long piece of code to add an **Addition** layer:

```python
add_node = gs.Node(op="Add", inputs=[x1, x2], outputs=[y])
graph.nodes.append(add_node)
```

You can simply write:

```python
y = graph.layer(op="Add", inputs=[x1, x2])

```

### **Example: Creating an Add Function Using `Graph.layer()`**

```python
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])
```

- **`@gs.Graph.register()`** → This registers the function `add` as a method of the `Graph` class.
- **`def add(self, a, b)`** → This function takes two tensors, `a` and `b`.
- **`self.layer(op="Add", inputs=[a, b], outputs=["add_out"])`** →
    - Creates an **Add** node.
    - Takes `a` and `b` as inputs.
    - Produces an output tensor named `"add_out"`.
    - Automatically **adds the node to the graph**.
    

> @gs.Graph.register() allows these functions to be used as built-in methods on gs.Graph.
>