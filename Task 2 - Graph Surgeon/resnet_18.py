import sys
import torch
import onnx
import onnx_graphsurgeon as gs
from torchvision import models

def export_resnet18_onnx(onnx_filename):
    model = models.resnet18(pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Model exported to {onnx_filename}")

def count_node_occurrence(node_type, onnx_filename, use_graphsurgeon):
    model = onnx.load(onnx_filename)
    
    if use_graphsurgeon:
        graph = gs.import_onnx(model)
        count = sum(1 for node in graph.nodes if str(node.op).lower() == str(node_type).lower())
    else:
        count = sum(1 for node in model.graph.node if str(node.op_type).lower() == str(node_type).lower())
    
    print(f"{node_type} node occurrence: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("The command should be in the below format:")
        print("\tpython file_name.py <NodeType> <ONNX_filename>")
        sys.exit(1)
    
    node_type = sys.argv[1]
    onnx_filename = sys.argv[2]
    
    export_resnet18_onnx(onnx_filename)

    choice = input("Use graphsurgeon? (yes/no): ").strip().lower()
    use_graphsurgeon = choice == "yes"
    
    count_node_occurrence(node_type, onnx_filename, use_graphsurgeon)