import torch
import torchvision

# Load the PyTorch model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Dummy input tensor for export
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True, opset_version=11)
