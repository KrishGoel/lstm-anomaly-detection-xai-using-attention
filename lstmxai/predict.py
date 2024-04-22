from .utils import LSTMModel
from torch.utils.data import DataLoader
import torch

def predict(model: LSTMModel, data_loader: DataLoader) -> tuple[list[int], list[float], list[int], list[tuple[float, float]], list[torch.Tensor]]:
    """
    Predict the labels for the data_loader using the model and return attention weights.

    Args:
        model (LSTMModel): The trained LSTM model.
        data_loader (DataLoader): The DataLoader instance.

    Returns:
        tuple[list[int], list[float], list[int], list[tuple[float, float]], list[torch.Tensor]]: A tuple containing:
            1. List of int predictions.
            2. List of float predictions (probabilities).
            3. List of int true labels.
            4. List of tuples containing the predicted value and the actual label.
            5. List of attention weight tensors for each instance.
    """
    
    model.eval()

    predictions = []
    float_predictions = []
    true_labels = []
    predictions_with_labels = []
    attention_weights_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs, attention_weights = model(inputs)

            predicted_prob = torch.sigmoid(outputs).item()
            predicted_label = 1 if predicted_prob >= 0.5 else 0

            label = labels.item()
            float_predictions.append(predicted_prob)
            predictions.append(predicted_label)
            true_labels.append(int(label))
            predictions_with_labels.append((predicted_prob, label))
            attention_weights_list.append(attention_weights)
        
    return predictions, float_predictions, true_labels, predictions_with_labels, attention_weights_list