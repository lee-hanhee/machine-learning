import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as torch_data
import tqdm

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transform)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True)
test_loader = torch_data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False)

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
    
input_size = 28 * 28
hidden_size = 100
num_classes = len(class_names)
model = MLP(input_size, hidden_size, num_classes)
print("Model Architecture:")
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    pbar.set_postfix({'loss':running_loss/i})
    
model.eval()

correct = 0
total = len(test_dataset)
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        predicted = torch.argmax(outputs.data, 1)
        correct += (predicted == labels).sum().item()
print(f'Accuracy (test): {100 * correct / total}%')