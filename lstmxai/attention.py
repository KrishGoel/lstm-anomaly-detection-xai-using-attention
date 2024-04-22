from .utils import LSTMModel
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def collect_attention_weights(model: LSTMModel, data_loader: DataLoader) -> list[torch.Tensor]:
    """
    Collects attention weights from the model for each batch of data in the data loader.

    Args:
        model (LSTMModel): The LSTM model that includes an attention mechanism.
        data_loader (DataLoader): DataLoader containing the dataset for evaluation.

    Returns:
        List[torch.Tensor]: A list of tensors containing the attention weights from each batch.
    """

    model.eval()
    attention_maps = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(model.device)
            _, attention_weights = model(inputs)
            attention_maps.extend(attention_weights.detach().cpu())  # Detach and move to CPU

    return attention_maps

def plot_attention_weights(attention_maps: list[torch.Tensor], index: int = 0) -> None:
    """
    Plots the attention weights for each tensor in the list.

    Args:
        attention_maps (List[torch.Tensor]): A list of tensors containing the attention weights.
        index (int): The index of the tensor to plot. Default is 0.
    """
    numpy_data = attention_maps[index].numpy()
    plt.plot(np.arange(140), numpy_data)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Tensor Data')
    plt.show()
    
    # for i, attn_weights in enumerate(attention_maps):
    #     numpy_data = attn_weights.numpy()
    #     plt.plot(np.arange(140), numpy_data)
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.title('Plot of Tensor Data')
    #     plt.show()
    
    return None