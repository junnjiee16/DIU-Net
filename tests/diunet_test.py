import torch
from diunet import DIUNet
torch.cuda.empty_cache()

model = DIUNet(out_channels=2, scale = 0.25)
print(f"CUDA: {torch.cuda.is_available()}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

test_data = torch.randn((1, 1, 512, 512))
with torch.no_grad():
    x = model(test_data)
    print(x.shape)
