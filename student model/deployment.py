import torch
from nafnet_tiny import get_nafnet_tiny  # adjust if your model file is different

model = get_nafnet_tiny()
model.load_state_dict(torch.load("student_epoch_110.pth", map_location='cpu'))
model.eval()

# Dummy input (adjust height/width to match your expected webcam input)
dummy_input = torch.randn(1, 3, 240, 320)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "student_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                  "output": {0: "batch_size", 2: "height", 3: "width"}},
    opset_version=11
)

print("✅ Exported to student_model.onnx")