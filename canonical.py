import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------
# Configurations
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
batch_size = 64
learning_rate = 1e-3

# -----------------------
# Data
# -----------------------
# Example using MNIST dataset
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Model Definition
# -----------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1} Training Loss: {train_loss / len(train_loader):.4f}")

    # -----------------------
    # Evaluation Loop
    # -----------------------
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval ]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} Eval Loss: {eval_loss / len(test_loader):.4f}, Accuracy: {acc:.2
