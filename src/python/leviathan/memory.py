import torch
import torch.nn as nn
import torch.optim as optim


class LiquidStateMachine(nn.Module):
    """
    Reservoir Readout: Decodes the high-dimensional phi_dot state
    into a meaningful output (e.g., target food vector).
    """

    def __init__(self, input_dim, output_dim=2, lr=0.01):
        super().__init__()
        self.readout = nn.Linear(input_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.last_loss = 0.0

    def forward(self, x):
        return self.readout(x)

    def train_step(self, x, target):
        self.optimizer.zero_grad()
        prediction = self.forward(x)
        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()
        return self.last_loss
