import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image
import glob

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

# Helper function to get max softmax probabilities
def get_max_softmax_probs(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    max_probs = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.softmax(outputs, dim=1)
            max_probs.append(softmax_outputs.max(dim=1)[0])  # Max probability for each image

    return torch.cat(max_probs).cpu().numpy()

# Helper function to get max softmax probabilities for ensemble
def get_ensemble_max_softmax_probs(ensemble, data_loader, device):
    max_probs = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            ensemble_outputs = torch.zeros(images.size(0), 10).to(device)

            # Average predictions from all ensemble models
            for network in ensemble:
                outputs = network(images)
                ensemble_outputs += outputs

            ensemble_outputs /= len(ensemble)
            softmax_outputs = torch.softmax(ensemble_outputs, dim=1)
            max_probs.append(softmax_outputs.max(dim=1)[0])  # Max probability for each image

    return torch.cat(max_probs).cpu().numpy()

# Calculate AUROC and AUPR
def calculate_auroc_aupr(ind_probs, ood_probs):
    # Labels: in-distribution = 1, OOD = 0
    labels = [1] * len(ind_probs) + [0] * len(ood_probs)
    scores = list(ind_probs) + list(ood_probs)
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # AUPR (in-distribution as the positive class)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    
    return auroc, aupr










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

# Helper function to print label and probability for each sample
def print_predictions(image_paths, predicted_labels, predicted_probs, model_type):
    for i, (label, prob) in enumerate(zip(predicted_labels, predicted_probs)):
        print(f'{i+1}th sample ({os.path.basename(image_paths[i])}) with {model_type}: label = {label}, probability = {prob:.4f}')

# Function to get predicted labels and probabilities
def get_labels_and_probs(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.softmax(outputs, dim=1)
            max_probs, labels = softmax_outputs.max(dim=1)  # Get max probability and corresponding label
            predicted_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(max_probs.cpu().numpy())

    return predicted_labels, predicted_probs

# Function to get labels and probabilities for ensemble
def get_ensemble_labels_and_probs(ensemble, data_loader, device):
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            ensemble_softmax_outputs = torch.zeros(images.size(0), 10).to(device)

            # Softmax predictions from each ensemble model
            for network in ensemble:
                outputs = network(images)
                softmax_outputs = torch.softmax(outputs, dim=1)
                ensemble_softmax_outputs += softmax_outputs

            # Average the softmax outputs
            ensemble_softmax_outputs /= len(ensemble)
            
            # Get max probability and corresponding label
            max_probs, labels = ensemble_softmax_outputs.max(dim=1)
            predicted_labels.extend(labels.cpu().numpy())
            predicted_probs.extend(max_probs.cpu().numpy())

    return predicted_labels, predicted_probs

# Updated test function for OOD detection on independent PNG images
def test_ood_images_on_model(image_folder):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to the saved models
    model_save_path = './trained_models/'

    # Load the single model
    single_model = SimpleCNN().to(device)
    single_model.load_state_dict(torch.load(os.path.join(model_save_path, 'single_model.pth')))
    single_model.eval()

    # Load the ensemble models
    num_ensembles = 5
    loaded_ensemble = [SimpleCNN().to(device) for _ in range(num_ensembles)]
    for idx, network in enumerate(loaded_ensemble):
        network.load_state_dict(torch.load(os.path.join(model_save_path, f'ensemble_model_{idx}.pth')))
        network.eval()

    # Define the same transforms as for MNIST (resize and normalize)
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    # Load and preprocess OOD images
    ood_images, image_paths = load_and_preprocess_ood_images(image_folder, transform, device)
    
    # Perform predictions on the OOD images using the single model
    predicted_labels_single, predicted_probs_single = get_labels_and_probs(single_model, [ood_images], device)
    print("Single Model predictions:")
    print_predictions(image_paths, predicted_labels_single, predicted_probs_single, "Single Model")
    
    # Perform predictions on the OOD images using the ensemble model
    predicted_labels_ensemble, predicted_probs_ensemble = get_ensemble_labels_and_probs(loaded_ensemble, [ood_images], device)
    print("\nEnsemble Model predictions:")
    print_predictions(image_paths, predicted_labels_ensemble, predicted_probs_ensemble, "Ensemble Model")

# Example usage:
# Provide the path to the folder containing OOD PNG images
test_ood_images_on_model('./pic/')