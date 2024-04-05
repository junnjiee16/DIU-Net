from torch.optim import Adam
from diunet import DIUNet

# get dataloaders and dataset


model = DIUNet()
optimizer = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
batch_size = 8
epochs = 120

for i in range(epochs):
    pass