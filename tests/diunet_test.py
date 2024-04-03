import torch
from diunet import DIUNet

model = DIUNet()
print(torch.cuda.is_available())
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

test_data = torch.randn((1, 1, 512, 512))
# with torch.no_grad():
#     x = model(test_data)
#     print(x.shape)
