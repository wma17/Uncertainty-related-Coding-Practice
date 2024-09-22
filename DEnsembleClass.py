import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Parameters
img_size = 28
num_label = 10
batch_size = 128
learning_rate = 0.05
momentum = 0.9
num_epochs = 30
dropout_prob = 0.5

# Define CNN architecture based on the guide
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_label):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Custom weight initialization function
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # kaiming initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Define ensemble of models with different initializations
num_ensembles = 5
networks = [SimpleCNN().to(device) for _ in range(num_ensembles)]

# Apply different initializations
for network in networks:
    network.apply(init_weights)

# Optimizers and loss function for each network
optimizers = [optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum) for network in networks]
criterion = nn.CrossEntropyLoss()


# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Train each model in the ensemble
        for idx, network in enumerate(networks):
            outputs = network(images)
            loss = criterion(outputs, labels)
            
            optimizers[idx].zero_grad()
            loss.backward()
            optimizers[idx].step()
            
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Testing the ensemble
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Average predictions from all models in the ensemble
        ensemble_outputs = torch.zeros(labels.size(0), num_label).to(device)
        for network in networks:
            ensemble_outputs += network(images)
        ensemble_outputs /= num_ensembles
        
        _, predicted = torch.max(ensemble_outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy of the ensemble: {100 * correct / total:.2f}%')


# Directory to save the models
model_save_path = './trained_models/'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Save the single model (choose any one model from the ensemble)
torch.save(networks[3].state_dict(), os.path.join(model_save_path, 'single_model.pth'))

# Save all the ensemble models
for idx, network in enumerate(networks):
    torch.save(network.state_dict(), os.path.join(model_save_path, f'ensemble_model_{idx}.pth'))

print("Models saved successfully.")