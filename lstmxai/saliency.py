from .utils import LSTMModel
import torch
from torch.utils.data import DataLoader
import numpy as np

def compute_saliency_map(model: LSTMModel, data_loader: DataLoader, index: int) -> np.ndarray:
    """
    Compute the saliency map of a given input with respect to the model's output

    Args:
        model (LSTMModel): The model to compute the saliency map for
        data_loader (DataLoader): The data loader containing the input data
        index (int): The index of the input data to compute the saliency map for

    Returns:
        np.ndarray: The saliency map
    """

    model.eval()

    if index < 0 or index >= len(data_loader.dataset):
        raise ValueError("Index is out of range")

    inputs, _ = data_loader.dataset[index]
    inputs = inputs.unsqueeze(0).to(model.device)
    inputs.requires_grad_()

    outputs, _ = model(inputs)
    outputs = torch.sigmoid(outputs)

    model.zero_grad()
    outputs[:, outputs.argmax(dim=1)].backward()

    saliency_map = inputs.grad.abs().max(dim=2)[0].squeeze().cpu().detach().numpy()

    return saliency_map

def compute_all_saliency_maps(model: LSTMModel, data_loader: DataLoader) -> list[np.ndarray]:
    """
    Compute the saliency maps for all instances in the data loader with respect to the model's output

    Args:
        model (LSTMModel): The model to compute the saliency maps for
        data_loader (DataLoader): The data loader containing the input data

    Returns:
        list[np.ndarray]: The list of saliency maps for all instances
    """

    saliency_maps = []

    for i in range(len(data_loader)):
        saliency_maps.append(compute_saliency_map(model, data_loader, i))
    
    return saliency_maps