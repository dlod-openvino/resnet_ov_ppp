from torchvision.models import resnet50, ResNet50_Weights
import torch

# https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights, progress=False).cpu().eval()

# define input and output node
dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
input_names,  output_names = ["images"], ['output']

torch.onnx.export(model, 
                 dummy_input, 
                 "resnet50.onnx", 
                 verbose=True, 
                 input_names=input_names, 
                 output_names=output_names,
                 opset_version=13 
                 )

