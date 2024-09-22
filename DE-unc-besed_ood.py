import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Define the CNN architecture (same as the one used in training)
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
        for data in data_loader:
            images = data[0].to(device)
            outputs = model(images)
            softmax_outputs = torch.softmax(outputs, dim=1)
            max_probs.append(softmax_outputs.max(dim=1)[0])  # Max probability for each image

    return torch.cat(max_probs).cpu().numpy()

# Helper function to get ensemble predictions
def get_ensemble_predictions(ensemble, data_loader, device):
    ensemble_preds = []

    with torch.no_grad():
        for data in data_loader:
            images = data[0].to(device)
            batch_preds = []

            # Collect predictions from each ensemble member
            for network in ensemble:
                outputs = network(images)
                softmax_outputs = torch.softmax(outputs, dim=1)
                batch_preds.append(softmax_outputs.unsqueeze(0))  # Add an extra dimension for stacking

            # Stack predictions from all ensemble members
            batch_preds = torch.cat(batch_preds, dim=0)  # Shape: [S, N, K]
            ensemble_preds.append(batch_preds)

    # Concatenate all batches
    ensemble_preds = torch.cat(ensemble_preds, dim=1)  # Shape: [S, Total_N, K]
    return ensemble_preds  # Returns predictions from all ensemble members

# Function to compute uncertainties
def compute_uncertainties(ensemble_preds):
    """
    ensemble_preds: Tensor of shape [S, N, K]
    S: Number of ensemble members
    N: Number of samples
    K: Number of classes
    """
    S, N, K = ensemble_preds.shape

    # Mean predictive distribution
    mean_preds = ensemble_preds.mean(dim=0)  # Shape: [N, K]

    # Total uncertainty (Predictive Entropy)
    total_uncertainty = -torch.sum(mean_preds * torch.log(mean_preds + 1e-10), dim=1)  # Shape: [N]

    # Expected Entropy (Aleatoric Uncertainty)
    entropies = -torch.sum(ensemble_preds * torch.log(ensemble_preds + 1e-10), dim=2)  # Shape: [S, N]
    aleatoric_uncertainty = entropies.mean(dim=0)  # Shape: [N]

    # Epistemic Uncertainty (Mutual Information)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty  # Shape: [N]

    return total_uncertainty.cpu().numpy(), aleatoric_uncertainty.cpu().numpy(), epistemic_uncertainty.cpu().numpy()

# Calculate AUROC and AUPR using uncertainties
def calculate_auroc_aupr_uncertainty(ind_uncertainty, ood_uncertainty):
    # Labels: in-distribution = 0, OOD = 1
    labels = [0] * len(ind_uncertainty) + [1] * len(ood_uncertainty)
    
    # Scores: higher uncertainty indicates higher likelihood of being OOD
    scores = np.concatenate([ind_uncertainty, ood_uncertainty])
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # AUPR (OOD as the positive class)
    aupr = average_precision_score(labels, scores, pos_label=1)
    
    return auroc, aupr

# Function to plot uncertainty histograms
def plot_uncertainty_histograms(ind_uncertainty, ood_uncertainty, title):
    plt.figure(figsize=(8, 6))
    plt.hist(ind_uncertainty, bins=50, alpha=0.5, label='In-distribution (MNIST)')
    plt.hist(ood_uncertainty, bins=50, alpha=0.5, label='OOD')
    plt.title(title)
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main testing function with uncertainty quantification
def test_ood_detection_with_uncertainty():
    # Set device
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

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

    # In-distribution dataset (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

   # Out-of-distribution dataset (Omniglot) with color inversion
    ood_transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match MNIST
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # Invert the colors
        transforms.Normalize((0.5,), (0.5,))
    ])
    ood_dataset = datasets.Omniglot(root='./data', background=False, transform=ood_transform, download=True)
    ood_loader = DataLoader(dataset=ood_dataset, batch_size=128, shuffle=False)


    # Out-of-distribution dataset (EMNIST)
    emnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    emnist_dataset = datasets.EMNIST(root='./data', split='letters', train=False, transform=emnist_transform, download=True)
    emnist_loader = DataLoader(dataset=emnist_dataset, batch_size=128, shuffle=False)

    # Out-of-distribution dataset (KMNIST)
    kmnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    kmnist_dataset = datasets.KMNIST(root='./data', train=False, transform=kmnist_transform, download=True)
    kmnist_loader = DataLoader(dataset=kmnist_dataset, batch_size=128, shuffle=False)

    # Get ensemble predictions for in-distribution and OOD datasets
    ind_ensemble_preds = get_ensemble_predictions(loaded_ensemble, test_loader, device)
    ood_ensemble_preds = get_ensemble_predictions(loaded_ensemble, ood_loader, device)
    emnist_ensemble_preds = get_ensemble_predictions(loaded_ensemble, emnist_loader, device)
    kmnist_ensemble_preds = get_ensemble_predictions(loaded_ensemble, kmnist_loader, device)

    # Compute uncertainties for in-distribution data
    ind_total_unc, ind_aleatoric_unc, ind_epistemic_unc = compute_uncertainties(ind_ensemble_preds)

    # Compute uncertainties for OOD data (Omniglot)
    ood_total_unc, ood_aleatoric_unc, ood_epistemic_unc = compute_uncertainties(ood_ensemble_preds)

    # Compute uncertainties for OOD data (EMNIST)
    emnist_total_unc, emnist_aleatoric_unc, emnist_epistemic_unc = compute_uncertainties(emnist_ensemble_preds)

    # Compute uncertainties for OOD data (KMNIST)
    kmnist_total_unc, kmnist_aleatoric_unc, kmnist_epistemic_unc = compute_uncertainties(kmnist_ensemble_preds)

    # Plot uncertainties for Omniglot
    plot_uncertainty_histograms(ind_total_unc, ood_total_unc, 'Total Uncertainty (Omniglot)')
    plot_uncertainty_histograms(ind_aleatoric_unc, ood_aleatoric_unc, 'Aleatoric Uncertainty (Omniglot)')
    plot_uncertainty_histograms(ind_epistemic_unc, ood_epistemic_unc, 'Epistemic Uncertainty (Omniglot)')

    # Plot uncertainties for EMNIST
    plot_uncertainty_histograms(ind_total_unc, emnist_total_unc, 'Total Uncertainty (EMNIST)')
    plot_uncertainty_histograms(ind_aleatoric_unc, emnist_aleatoric_unc, 'Aleatoric Uncertainty (EMNIST)')
    plot_uncertainty_histograms(ind_epistemic_unc, emnist_epistemic_unc, 'Epistemic Uncertainty (EMNIST)')

    # Plot uncertainties for KMNIST
    plot_uncertainty_histograms(ind_total_unc, kmnist_total_unc, 'Total Uncertainty (KMNIST)')
    plot_uncertainty_histograms(ind_aleatoric_unc, kmnist_aleatoric_unc, 'Aleatoric Uncertainty (KMNIST)')
    plot_uncertainty_histograms(ind_epistemic_unc, kmnist_epistemic_unc, 'Epistemic Uncertainty (KMNIST)')

    # Optionally, compute AUROC and AUPR using uncertainties for Omniglot
    auroc_total_omniglot, aupr_total_omniglot = calculate_auroc_aupr_uncertainty(ind_total_unc, ood_total_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Total Uncertainty, Omniglot): {auroc_total_omniglot:.4f}, AUPR: {aupr_total_omniglot:.4f}')
    auroc_aleatoric_omniglot, aupr_aleatoric_omniglot = calculate_auroc_aupr_uncertainty(ind_aleatoric_unc, ood_aleatoric_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, Omniglot): {auroc_aleatoric_omniglot:.4f}, AUPR: {aupr_aleatoric_omniglot:.4f}')
    auroc_epistemic_omniglot, aupr_epistemic_omniglot = calculate_auroc_aupr_uncertainty(ind_epistemic_unc, ood_epistemic_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, Omniglot): {auroc_epistemic_omniglot:.4f}, AUPR: {aupr_epistemic_omniglot:.4f}')

    # Similarly for EMNIST
    auroc_total_emnist, aupr_total_emnist = calculate_auroc_aupr_uncertainty(ind_total_unc, emnist_total_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Total Uncertainty, EMNIST): {auroc_total_emnist:.4f}, AUPR: {aupr_total_emnist:.4f}')
    auroc_aleatoric_emnist, aupr_aleatoric_emnist = calculate_auroc_aupr_uncertainty(ind_aleatoric_unc, emnist_aleatoric_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, EMNIST): {auroc_aleatoric_emnist:.4f}, AUPR: {aupr_aleatoric_emnist:.4f}')
    auroc_epistemic_emnist, aupr_epistemic_emnist = calculate_auroc_aupr_uncertainty(ind_epistemic_unc, emnist_epistemic_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, EMNIST): {auroc_epistemic_emnist:.4f}, AUPR: {aupr_epistemic_emnist:.4f}')

    # Similarly for KMNIST
    auroc_total_kmnist, aupr_total_kmnist = calculate_auroc_aupr_uncertainty(ind_total_unc, kmnist_total_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Total Uncertainty, KMNIST): {auroc_total_kmnist:.4f}, AUPR: {aupr_total_kmnist:.4f}')
    auroc_aleatoric_kmnist, aupr_aleatoric_kmnist = calculate_auroc_aupr_uncertainty(ind_aleatoric_unc, kmnist_aleatoric_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, KMNIST): {auroc_aleatoric_kmnist:.4f}, AUPR: {aupr_aleatoric_kmnist:.4f}')
    auroc_epistemic_kmnist, aupr_epistemic_kmnist = calculate_auroc_aupr_uncertainty(ind_epistemic_unc, kmnist_epistemic_unc)
    print(f'Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, KMNIST): {auroc_epistemic_kmnist:.4f}, AUPR: {aupr_epistemic_kmnist:.4f}')

if __name__ == "__main__":
    test_ood_detection_with_uncertainty()