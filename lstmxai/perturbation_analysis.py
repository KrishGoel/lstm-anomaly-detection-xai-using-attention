from .utils import LSTMModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def perturbe_data(data_loader: DataLoader, noise_factor: float = 0.1) -> DataLoader:
    """
    Perturbs the data by adding noise to the input features.

    Args:
        data_loader (DataLoader): The DataLoader instance containing the data.
        noise_factor (float): The factor controlling the amount of noise to be added.

    Returns:
        DataLoader: The DataLoader instance with perturbed data.
    """

    perturbed_samples = []
    for X, y in data_loader:
        noise = torch.randn_like(X) * noise_factor
        X_perturbed = X + noise
        perturbed_samples.append((X_perturbed, y))
    
    perturbed_data_loader = DataLoader(perturbed_samples, batch_size=data_loader.batch_size)
    return perturbed_data_loader

def perturbation_analysis(model: LSTMModel, data_loader: DataLoader, noise_factor: float = 0.1, index: int = 0) -> tuple[float, float, list[float]]:
    """
    Performs perturbation analysis on the model using the given data loader for a specific data row.

    Args:
        model (LSTMModel): The trained model to be analyzed.
        data_loader (DataLoader): The DataLoader instance containing the data.
        noise_factor (float): The factor controlling the amount of noise to be added.
        index (int): Index of the data row to analyze.

    Returns:
        tuple[float, float, list[float]]: The original loss, perturbed loss and list of float sensitivity values.
    """
    
    model.eval()

    perturbed_data_loader = perturbe_data(data_loader, noise_factor)

    original_loss = 0.0
    perturbed_loss = 0.0
    sensitivity_values = []

    with torch.no_grad():
        for i, (original_X, original_y) in enumerate(data_loader):
            if i == index:
                for j in range(len(original_X[0])):
                    perturbed_X, _ = next(iter(perturbed_data_loader))
                    perturbed_X = perturbed_X.squeeze(0)

                    original_outputs = model(original_X)
                    perturbed_outputs = model(perturbed_X)

                    if isinstance(original_outputs, tuple):
                        original_outputs = original_outputs[0]
                    if isinstance(perturbed_outputs, tuple):
                        perturbed_outputs = perturbed_outputs[0]

                    original_point_loss = nn.MSELoss()(original_outputs.squeeze(0), original_y)
                    perturbed_point_loss = nn.MSELoss()(perturbed_outputs.squeeze(0), original_y)


                    original_loss += original_point_loss.item()
                    perturbed_loss += perturbed_point_loss.item()

                    point_sensitivity = torch.abs(original_point_loss - perturbed_point_loss) / noise_factor
                    sensitivity_values.append(point_sensitivity.item())

                break

    original_loss /= len(sensitivity_values)
    perturbed_loss /= len(sensitivity_values)

    return original_loss, perturbed_loss, sensitivity_values
