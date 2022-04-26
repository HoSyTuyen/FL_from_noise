import torch.nn as nn

class latent_mapping(nn.Module):
    def __init__(self, inputDim=100):
        super(latent_mapping, self).__init__()

        self.Flatten = nn.Flatten()
        self.fc1 = nn.Linear(inputDim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, inputDim)

    def forward(self, z):
        z_op = self.Flatten(z)
        z_op = self.fc1(z_op)
        z_op = self.bn(z_op)
        z_op = self.relu(z_op)
        z_op = self.fc2(z_op)
        z_op = self.output(z_op)

        return z_op