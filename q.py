# Create a Python file with basic PyTorch questions and solutions

# basic_pytorch_questions.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. Create a PyTorch tensor of shape (3, 4) filled with random numbers
tensor1 = torch.rand(3, 4)

# 2. Convert a NumPy array to a PyTorch tensor and back
np_array = np.array([[1, 2], [3, 4]])
tensor2 = torch.from_numpy(np_array)
np_array_back = tensor2.numpy()

# 3. Perform matrix multiplication between two tensors
a = torch.rand(2, 3)
b = torch.rand(3, 2)
matmul_result = torch.matmul(a, b)

# 4. Reshape a tensor from shape (4, 3) to (2, 2, 3)
tensor3 = torch.rand(4, 3)
reshaped = tensor3.view(2, 2, 3)

# 5. Check if GPU is available and move tensor to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor4 = torch.rand(2, 2).to(device)

# 6. Create a tensor with gradient tracking enabled
x = torch.tensor([2.0], requires_grad=True)

# 7. Compute gradients using autograd for y = x^2
y = x ** 2
y.backward()
gradient = x.grad

# 8. Freeze model parameters during fine-tuning
model = nn.Linear(10, 1)
for param in model.parameters():
    param.requires_grad = False

# 9. Toggle between training and evaluation mode
model.train()
model.eval()

# 10. Disable gradient computation
with torch.no_grad():
    inference = model(torch.rand(1, 10))

# 11. Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

net = Net()

# 12. Use DataLoader to load a toy dataset
x_data = torch.rand(100, 10)
y_data = torch.rand(100, 1)
dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# 13. Canonical training loop
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(1):
    net.train()
    for x_batch, y_batch in loader:
        # FWD Propagation
        preds = net(x_batch)
        loss = loss_fn(preds, y_batch)
        
        # BWD Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# 13.5 Eval mode
net.eval()

correct = 0
total = len(dataset)
with torch.no_grad():
    for x_batch, y_batch in loader:
        preds = net(x_batch)
        loss = loss_fn(preds, y_batch)
        # Assuming binary classification for simplicity
        predicted = (preds > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")

# 14. Save and load a model
torch.save(net.state_dict(), "model.pth")
net.load_state_dict(torch.load("model.pth"))

# 15. Custom loss function
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)

# 16. Concatenate tensors
concat = torch.cat([torch.rand(2, 3), torch.rand(2, 3)], dim=0)

# 17. Stack tensors
stack = torch.stack([torch.rand(2, 3), torch.rand(2, 3)], dim=0)

# 18. Detach vs. no_grad
a = torch.tensor([3.0], requires_grad=True)
b = a.detach()  # no grad tracking
with torch.no_grad():
    c = a + 2  # also no grad tracking

# 19. Manual ReLU
x = torch.tensor([-1.0, 2.0])
relu_manual = x * (x > 0).float()

# 20. Dropout
dropout = nn.Dropout(p=0.5)
dropout_output = dropout(torch.rand(5))

# 21. Transform 
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# 22. MLP 
class MLP(nn.Module):
    """Multilayer Perceptron (MLP) model."""

    def __init__(self, input_size:int, hidden_size:int, num_classes:int):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.predict = nn.Linear(hidden_size, num_classes)

    def get_embedding(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x
    
# 23. CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, 10)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out
    
# Move to device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)