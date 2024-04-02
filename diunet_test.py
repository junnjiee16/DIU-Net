import torch
from diunet.diunet import DIUNet

model = DIUNet()
test_data = torch.randn((32, 1, 512, 512))
with torch.no_grad():
    x = model(test_data)
    print(x.shape)
