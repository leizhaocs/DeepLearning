import torch.nn as nn
import torch.nn.functional as F

# the deep q network
class DQN(nn.Module):

    # constructor
    def __init__(self, seq_length, img_height, img_width, num_actions):

        super().__init__()

        self.conv1 = nn.Conv2d(seq_length, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convh = conv2d_size_out(conv2d_size_out(img_height, 8, 4), 4, 2)
        convw = conv2d_size_out(conv2d_size_out(img_width, 8, 4), 4, 2)

        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.out = nn.Linear(256, num_actions)

    # forwad propagate
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.out(x)
        return x
