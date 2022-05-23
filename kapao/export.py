import torch

from models.experimental import attempt_load

model = attempt_load("./weights/kapao_s_coco.pt", map_location="cpu")

x = torch.randn(1, 3, 768, 768, requires_grad=True)
torch.onnx.export(
    model,
    x,
    "./export.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    do_constant_folding=True,
    dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                  'output': {0: 'batch_size'}}
)
