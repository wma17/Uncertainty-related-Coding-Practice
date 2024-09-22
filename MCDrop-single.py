import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image
import glob

# Define CNN architecture with dropout
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Helper function to perform MC Dropout and get predictions
def get_mc_dropout_predictions(model, data_loader, device, mc_iterations):
    model.train()  # Enable dropout during inference

    
    all_preds = []
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            # Tensor to store MC predictions [mc_iterations, batch_size, num_classes]
            mc_preds = torch.zeros(mc_iterations, images.size(0), 10).to(device)
            
            for i in range(mc_iterations):
                outputs = model(images)
                softmax_outputs = torch.softmax(outputs, dim=1)
                mc_preds[i] = softmax_outputs
            
            all_preds.append(mc_preds)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=1)  # Shape: [mc_iterations, Total_N, num_classes]
    return all_preds

# Helper function to load and preprocess OOD images (png files)
def load_and_preprocess_ood_images(image_folder, transform, device):
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))  # Load all PNG images
    ood_images = []
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_tensor = transform(img)  # Apply the same transforms as MNIST (resize, normalize, etc.)
        ood_images.append(img_tensor)
    
    # Stack all images into a single tensor
    ood_images = torch.stack(ood_images).to(device)
    return ood_images, image_paths

# Function to compute final labels and probabilities from MC predictions
def compute_final_predictions(mc_preds):
    """
    mc_preds: Tensor of shape [mc_iterations, batch_size, num_classes]
    Returns:
    predicted_labels: List of predicted labels
    predicted_probs: List of predicted probabilities
    """
    # Mean of the predictions over the MC iterations
    mean_preds = mc_preds.mean(dim=0)  # Shape: [batch_size, num_classes]
    
    # Get the predicted labels and their corresponding max probabilities
    max_probs, predicted_labels = torch.max(mean_preds, dim=1)  # Get max probability and corresponding label
    
    return predicted_labels.cpu().numpy(), max_probs.cpu().numpy()

# Helper function to print label and probability for each sample
def print_predictions(image_paths, predicted_labels, predicted_probs):
    print("MC-Dropout Model predictions:")
    for i, (label, prob) in enumerate(zip(predicted_labels, predicted_probs)):
        print(f'{i+1}th sample ({os.path.basename(image_paths[i])}) with MC-Dropout Model: label = {label}, probability = {prob:.4f}')

# Updated test function for OOD detection using MC Dropout
def test_ood_images_mc_dropout(image_folder, mc_iterations=10):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to the saved models
    model_save_path = './trained_models/'

    # Load the single model with Dropout and BatchNorm
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'mc_dropout_model.pth')))
    
    # Define the same transforms as for MNIST (resize and normalize)
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # Load and preprocess OOD images
    ood_images, image_paths = load_and_preprocess_ood_images(image_folder, transform, device)
    
    # Perform MC Dropout predictions on the OOD images
    mc_preds = get_mc_dropout_predictions(model, [ood_images], device, mc_iterations)
    
    # Compute final predicted labels and probabilities from MC Dropout predictions
    predicted_labels, predicted_probs = compute_final_predictions(mc_preds)
    
    # Print predictions for each sample
    print_predictions(image_paths, predicted_labels, predicted_probs)

# Example usage:
# Provide the path to the folder containing OOD PNG images
test_ood_images_mc_dropout('./pic/')