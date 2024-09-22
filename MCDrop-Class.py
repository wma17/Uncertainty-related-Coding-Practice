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

# Define CNN architecture with dropout
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

# Initialize the model
model = SimpleCNN().to(device)
model.apply(init_weights)

# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
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
    model.train()  # Ensure dropout is enabled during training
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing with MC-dropout
def mc_dropout_predict(model, images, mc_iterations):
    model.train()  # Enable dropout during inference
    outputs = torch.zeros(mc_iterations, images.size(0), num_label).to(device)
    with torch.no_grad():
        for i in range(mc_iterations):
            outputs[i] = model(images)
    return outputs

# Evaluate the model using MC-dropout
model.eval()  # Set to evaluation mode to deactivate other layers like BatchNorm
correct = 0
total = 0
mc_iterations = 10  # Number of MC samples

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Get MC-dropout outputs
        mc_outputs = mc_dropout_predict(model, images, mc_iterations)
        
        # Average over MC iterations
        mean_output = mc_outputs.mean(dim=0)  # Shape: [batch_size, num_label]
        
        # Predicted class
        _, predicted = torch.max(mean_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy with MC-dropout: {100 * correct / total:.2f}%')

# Save the model
model_save_path = './trained_models/'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

torch.save(model.state_dict(), os.path.join(model_save_path, 'mc_dropout_model.pth'))

print("Model saved successfully.")