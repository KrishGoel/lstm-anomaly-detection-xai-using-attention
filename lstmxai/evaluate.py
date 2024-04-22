from .utils import LSTMModel
from torch.utils.data import DataLoader
from .predict import predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_LSTM(model: LSTMModel, data_loader: DataLoader) -> dict:
    """
    Evaluate the LSTM model using the data_loader.
    
    Args:
        model (LSTMModel): The trained LSTM model.
        data_loader (DataLoader): The DataLoader instance.
        
    Returns:
        dict: A dictionary containing the performance metrics (accuracy, precision, recall, f1).     
    """

    model.eval()
    all_predictions, _, true_labels, _, _ = predict(model, data_loader)

    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions)
    recall = recall_score(true_labels, all_predictions)
    f1 = f1_score(true_labels, all_predictions)
    
    performance_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return performance_metrics