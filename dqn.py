import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_channels, action_size):
        super(QNetwork, self).__init__()
        # Suppose the input is (batch_size, input_channel, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4) # (batch_size, 32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # (batch_size, 64, 9, 9)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)