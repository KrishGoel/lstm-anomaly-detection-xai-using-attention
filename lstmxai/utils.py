import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

default_LSTM_config = {
    "input_dim": 1,
    "hidden_dim": 80,
    "num_layers": 3,
    "output_dim": 1,
    "device": torch.device('cpu')
}

def raw_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the raw data from the file.

    Args:
        file_path (str): The path to the file.

    Returns:
        tuple: A tuple containing the data and labels as numpy.ndarray.
    """

    df = pd.read_csv(file_path, header=None)

    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    return data, labels


def load_and_preprocess(file_path: str, test_size: float = 0.3, batch_size: int = 10) -> tuple[DataLoader, DataLoader]:
    """
    Load and preprocess the data from the file_path.

    Args:
        file_path (str): The path to the file.
        test_size (float): The size of the test set. Default is 0.3.
        batch_size (int): The batch size for the DataLoader. Default is 10.

    Returns:
        tuple: A tuple containing the training and testing DataLoader instances.
    """

    X, y = raw_data(file_path)

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Split data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    train_X = train_X.unsqueeze(-1)  # Add an extra dimension for LSTM input
    test_X = test_X.unsqueeze(-1)    # Add an extra dimension for LSTM input

    class ECGDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    # Create Dataset instances
    train_dataset = ECGDataset(train_X, train_y)
    test_dataset = ECGDataset(test_X, test_y)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader

class Attention(nn.Module):
    """
    Attention Layer for LSTM Model.

    Args:
        hidden_size (int): The number of hidden units in the LSTM.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Initialize attention weights and a small fully connected layer
        self.attention_fc = nn.Linear(hidden_size, 1)
    
    def forward(self, outputs):
        # Apply a small network over the last dimension
        scores = self.attention_fc(outputs).squeeze(2)  # (batch_size, seq_length)
        alpha = F.softmax(scores, dim=1)               # Softmax over sequence
        # Weighted sum of lstm outputs, based on attention weights
        context = torch.bmm(alpha.unsqueeze(1), outputs).squeeze(1)
        return context, alpha

class LSTMModel(nn.Module):
    """
    LSTM Model for ECG Classification.

    Args:
        input_dim (int): The number of input dimensions. Default is 1.
        hidden_dim (int): The number of hidden units in the LSTM. Default is 100.
        num_layers (int): The number of LSTM layers. Default is 1.
        output_dim (int): The number of output dimensions. Default is 1.
        device (torch.device): The device (CPU or GPU) to use. Default is 'cpu'.
    """
    
    def __init__(self, input_dim: int = default_LSTM_config['input_dim'], hidden_dim: int = default_LSTM_config['hidden_dim'], num_layers: int = default_LSTM_config["num_layers"], output_dim: int = default_LSTM_config['output_dim'], device: torch.device = default_LSTM_config['device']):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(device)
        self.attention = Attention(hidden_dim).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)
        self.config = {
            'Input Layer Dimensions': input_dim,
            'Hidden Layer Dimensions': hidden_dim,
            'Number of Layers:': num_layers,
            'Output Dimensions': output_dim,
            'Model on device': device
        }
        print("LSTM Model Config:", self.config)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out, attention_weights = self.attention(lstm_out)
        return self.fc(attention_out), attention_weights

def save_model(model: LSTMModel, model_path: str) -> str:
    """
    Save the model to the model_path.

    Args:
        model: The model to save.
        model_path (str): The path to save the model.

    Returns:
        model_path (str): The path to the saved model.
    """

    torch.save(model.state_dict(), model_path)
    return model_path

def load_model(model_path: str, input_dim: int = default_LSTM_config['input_dim'], hidden_dim: int = default_LSTM_config['hidden_dim'], num_layers: int = default_LSTM_config["num_layers"], output_dim: int = default_LSTM_config['output_dim'], device: torch.device = default_LSTM_config['device']):
    """
    Load the model from the model_path.

    Args:
        model_path (str): The path to the model.
        input_dim (int): The number of input dimensions. Default is 1.
        hidden_dim (int): The number of hidden units in the LSTM. Default is 100.
        num_layers (int): The number of LSTM layers. Default is 1.
        output_dim (int): The number of output dimensions. Default is 1.
        device (torch.device): The device (CPU or GPU) to use. Default is 'cpu'.

    Returns:
        model (LSTMModel): The loaded LSTM model.
    """

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model

def plot_ecg(data: pd.DataFrame, labels: pd.DataFrame, index: int = 0):
    """
    Plot the ECG signal.

    Args:
        data (torch.Tensor): The ECG signal data.
        index (int): Index of the signal to plot.
    """

    if index < 0 or index >= len(data):
        print("Invalid ECG Signal index.")
    
    else:
        plt.figure()
        plt.plot(data[index])
        plt.title(f"ECG Signal {index+1}/{len(data)} [{'Normal' if int(labels[index]) else 'Anomalous'}]")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    
def normalize_array(array: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """
    Normalize an array by subtracting the mean and dividing by the standard deviation.

    Args:
        array (torch.Tensor, np.ndarray, list): The input array to be normalized.

    Returns:
        np.ndarray: The normalized array.

    Raises:
        ValueError: If the input array is not of type torch.Tensor, np.ndarray, or list.
    """
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    elif isinstance(array, list):
        array = np.array(array)
    elif not isinstance(array, np.ndarray):
        raise ValueError("Unsupported data type. Please provide a torch.Tensor, np.ndarray, or list.")

    normalized_array = (array - np.mean(array)) / np.std(array)
    return normalized_array