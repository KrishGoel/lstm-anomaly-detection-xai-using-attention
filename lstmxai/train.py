from .utils import LSTMModel, save_model
from .evaluate import evaluate_LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

def train_LSTM(model: LSTMModel, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.modules.loss._Loss = nn.MSELoss(), learning_rate: float = 0.001, n_epochs: int = 150, visualize_loss: bool = True) -> str:
    """
    Train the LSTM model using the training data and potentially visualize the attention weights.

    Args:
        model (LSTMModel): The LSTM model to train.
        train_loader (DataLoader): The DataLoader instance for the training data.
        test_loader (DataLoader): The DataLoader instance for test data and getting performance metrics.
        criterion (nn.modules.loss._Loss): The loss function. Default is nn.MSELoss().
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.
        n_epochs (int): The number of epochs to train the model. Default is 70.
        visualize_loss (bool): Whether to visualize the training loss. Default is True.

    Returns:
        str: The path to the saved model.
    """
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            optimizer.zero_grad()
            
            # The model now returns outputs and attention weights
            outputs, attn_weights = model(inputs)
            
            # Make sure labels have the correct shape
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)
        train_losses.append(average_train_loss)
        print(f'Epoch {epoch+1}/{n_epochs}, Average Loss: {average_train_loss:.4f}')
    
    if visualize_loss:
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    metrics = evaluate_LSTM(model, test_loader)
    accuracy = round(metrics["accuracy"], 4)

    return save_model(model, f'./models/LSTM_{accuracy}.pth')