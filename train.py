from torch.optim import Adam
from diunet import DIUNet

# get dataloaders and dataset


model = DIUNet()
optimizer = Adam(model.parameters(), lr=1e-5)
epochs = 50