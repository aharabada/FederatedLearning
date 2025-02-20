import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(8*8*128, 1024)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc3 = torch.nn.Linear(512, 128)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(128, 32)
        self.dropout4 = torch.nn.Dropout(0.2)
        self.fc5 = torch.nn.Linear(32, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 8*8*128)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x