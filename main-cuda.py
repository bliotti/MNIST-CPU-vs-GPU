import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1. PREPARE THE DATA
# ----------------------------------------------------
# Transform to normalize the data and convert to PyTorch tensors
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Mean and Std for MNIST
    ]
)

# Download and create datasets
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. DEFINE THE MODEL
# ----------------------------------------------------
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         # A very simple feed-forward network
#         self.layer1 = nn.Linear(28 * 28, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, 10)

#     def forward(self, x):
#         # Flatten images into vectors
#         x = x.view(-1, 28 * 28)
#         x = nn.functional.relu(self.layer1(x))
#         x = nn.functional.relu(self.layer2(x))
#         x = self.layer3(x)
#         return x


# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(28*28, 128)
#         self.dropout1 = nn.Dropout(p=0.2)  # 20% dropout

#         self.layer2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout(p=0.2)

#         self.layer3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = F.relu(self.layer1(x))
#         x = self.dropout1(x)

#         x = F.relu(self.layer2(x))
#         x = self.dropout2(x)

#         x = self.layer3(x)
#         return x

# model = SimpleNet()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64 feature maps * 14 * 14 = 12544
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))  # => [batch_size, 32, 28, 28]
        x = self.pool(F.relu(self.conv2(x)))
        # => [batch_size, 64, 14, 14]

        x = x.view(x.size(0), -1)  # flatten => [batch_size, 64*14*14]
        x = F.relu(self.fc1(x))  # => [batch_size, 128]
        x = self.fc2(x)  # => [batch_size, 10]
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet().to(device)


# 3. SPECIFY LOSS FUNCTION AND OPTIMIZER
# ----------------------------------------------------
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam(model.parameters(), lr=5e-4)

# 4. TRAIN THE MODEL
# ----------------------------------------------------
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 5. EVALUATE THE MODEL
# ----------------------------------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")
