from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
from typing import Dict

def initialize_metrics(device) -> Dict:
    return {
        'GeneralizedDice': DiceScore(num_classes=2).to(device),
        'IoU': BinaryJaccardIndex().to(device),
        'Precision': BinaryPrecision().to(device),
        'Recall': BinaryRecall().to(device),
        'F1': BinaryF1Score().to(device)
    }

def reset_metrics(metrics_dict) -> None:
    """Reset all metrics for the next epoch."""
    for metrics in metrics_dict.values():
        for metric in metrics.values():
            metric.reset()

def log_metrics(writer, epoch: float, train_loss: float, val_loss: float, train_metrics: Dict, val_metrics: Dict) -> None:
    """Log metrics to TensorBoard and console."""
    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    for name in train_metrics:
        writer.add_scalars(name, {'train': train_metrics[name], 'val': val_metrics[name]}, epoch)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Dice: {train_metrics['GeneralizedDice']:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Dice: {val_metrics['GeneralizedDice']:.4f}")

