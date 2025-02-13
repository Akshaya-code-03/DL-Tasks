# ONNX

### What is ONNX?

- ONNX ( Open Neural Network Exchange ) - an open standard for exchanging deep learning models.
- It is an intermediary machine learning framework to convert between different machine learning frameworks
- It allows models trained in **one framework (PyTorch, TensorFlow, etc.)** to be used in another (ONNX Runtime, TensorFlow, etc.).

![image.png](image%203.png)

---

### **Why Use ONNX?**

- **Interoperability**: Convert models across different frameworks.
- **Optimized Execution**: ONNX models can run on different hardware (CPU, GPU, FPGA, etc.).
- **Standardization**: Provides a common format to avoid vendor lock-in.

---

![image.png](image%204.png)

---

![image.png](image%205.png)

---

### **ONNX Workflow:**

**Train Model in PyTorch/TensorFlow** 🡲 **Convert to ONNX** 🡲 **Run in ONNX Runtime**

---

### **1️⃣ Saving a File in ONNX Format (torch.onnx.export)**

**🔹 Purpose:**

Converts a framework to ONNX format and save it to a file.

**🔹 Function:**

`torch.onnx.export(model, dummy_input, filename, **kwargs)`

| **Parameter** | **Description** |
| --- | --- |
| `model` | The PyTorch model to be exported |
| `dummy_input` | A sample input tensor that defines input shape  |
| `filename` | The output file name |
| `input_names` | List of input tensor names for ONNX graph |
| `output_names` | List of output tensor names for ONNX graph |
| `dynamic_axes` | Specifies batch size flexibility  |

### 2️⃣ **Loading the ONNX File (onnx.load)**

**🔹Purpose:**

Load an ONNX model into memory for validation or further processing.

🔹 **Function:**

`onnx.load(filename)`

| **Parameter** | **Description** |
| --- | --- |
| `filename` | The path to the ONNX file  |

### **3️⃣ Simplifying the ONNX File (onnxsim.simplify)**

### **🔹Purpose:**

Optimize the ONNX model by **removing redundant layers, fusing operations, and reducing computations**.

### **🔹 Function:**

`onnxsim.simplify(onnx_model)`

| **Parameter** | **Description** |
| --- | --- |
| `onnx_model` | The loaded ONNX model object to be simplified. |

---

### **What Are Folds in ONNX?**

In ONNX, **folding** refers to a technique called **Constant Folding**, where operations that involve only constants are precomputed at the model conversion stage. This optimization reduces computational overhead during inference by replacing these operations with their computed results.

**Example:**

**Before Folding:**

`Y = X * 2 + 3`

Here, `2` and `3` are constants. During inference, every input `X` requires multiplication and addition operations.

**After Folding:**

`Y = X * precomputed_value`

Instead of storing * 2 and + 3 as separate operations, the compiler simplifies it to:

This reduces the number of operations needed at runtime.

---