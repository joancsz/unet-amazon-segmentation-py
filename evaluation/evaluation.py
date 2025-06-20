import torch
import numpy as np
import os
from sklearn.metrics import (confusion_matrix,
                            classification_report,
                            roc_curve,
                            auc,
                            precision_recall_curve,
                            average_precision_score)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import random

def evaluate_model(model, val_loader, save_path=None):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            preds = torch.sigmoid(model(imgs)).cpu().numpy()
            masks = masks.numpy()

            y_true.extend(masks.flatten())
            y_pred.extend((preds > 0.5).astype(np.uint8).flatten())
            y_score.extend(preds.flatten())

    # Classification report
    print(classification_report(y_true, y_pred, target_names=["Background", "Forest"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    labels = ['Background', 'Forest']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"), bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "ROC_curve.png"), bbox_inches='tight')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "precision_recall_curve.png"), bbox_inches='tight')
    plt.show()

def visualize_predictions_v2(model, dataset, num_samples=3, random_seed=None, save_path=None):
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_tensor = img.unsqueeze(0).cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor)).cpu().numpy()[0, 0]

        # Convert Sentinel-2 Bands 4 (NIR), 3 (Red), 2 (Green) to Visible RGB
        img_rgb = img[[2, 1, 0]].permute(1, 2, 0).cpu().numpy()
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())  # Min-Max Normalization
        img_rgb = np.clip(img_rgb ** 0.8, 0, 1)  # Apply Gamma Correction

        # Convert Ground Truth to 0-1 range
        mask = mask.squeeze().cpu().numpy()

        # Convert Prediction to Binary Mask
        pred_bin = (pred > 0.5).astype(np.uint8)

        # Create Error Map
        false_positive = (pred_bin == 1) & (mask == 0)  # FP (Background misclassified as Forest)
        false_negative = (pred_bin == 0) & (mask == 1)  # FN (Forest misclassified as Background)
        true_positive = (pred_bin == 1) & (mask == 1)  # TP (Correct Forest)
        true_negative = (pred_bin == 0) & (mask == 0)  # TN (Correct Background)

        error_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        error_map[true_positive] = [255, 255, 255]  # White for TP
        error_map[true_negative] = [0, 0, 0]        # Black for TN
        error_map[false_positive] = [255, 0, 0]     # Red for FP
        error_map[false_negative] = [0, 0, 255]     # Blue for FN

        # Plot Input Image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Input Image (RGB - Bands 4,3,2)")
        axes[i, 0].axis("off")

        # Plot Ground Truth
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth (Forest Mask)")
        axes[i, 1].axis("off")

        # Plot Predicted Mask
        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

        # Plot Error Map
        axes[i, 3].imshow(error_map)
        axes[i, 3].set_title("Error Map (FP: Red, FN: Blue)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_training_curves(train_metrics_list, val_metrics_list, metric_names=None,
                         figsize=(15, 10), smooth_factor=0.2, best_marker=True,
                         title=None, save_path=None):
    """
    Plot training/validation metrics with smoothing and annotations.

    Args:
        train_metrics_list: List of dicts (per epoch) of training metrics
        val_metrics_list: List of dicts (per epoch) of validation metrics
        metric_names: Which metrics to plot (None=plot all)
        smooth_factor: Exponential moving average smoothing (0-1)
        best_marker: Whether to mark best validation values
        title: Plot title
        save_path: If provided, saves plot to this path
    """
    if metric_names is None:
        metric_names = [k for k in train_metrics_list[0].keys() if k != 'loss']

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics + 1, 1, figsize=figsize, sharex=True)
    if n_metrics == 0:
        axes = [axes]

    # Extract loss and metrics
    train_loss = [e['loss'] for e in train_metrics_list]
    val_loss = [e['loss'] for e in val_metrics_list]

    # Apply smoothing
    def smooth(scalars, weight=0.6):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Plot Loss
    ax = axes[0]
    #ax.plot(train_loss, label='Train', alpha=0.3, color='tab:blue')
    #ax.plot(smooth(train_loss, smooth_factor), label='Train (smoothed)', color='tab:blue')
    ax.plot(train_loss, label=f'Train (min: {np.min(train_loss):.3f})', color='tab:blue')
    #ax.plot(val_loss, label='Val', alpha=0.3, color='tab:orange')
    ax.plot(val_loss, label=f'Val (min: {np.min(val_loss):.3f})', color='tab:orange')
    #ax.plot(smooth(val_loss, smooth_factor), label='Val (smoothed)', color='tab:orange')

    best_epoch = np.argmin(val_loss)
    best_value = val_loss[best_epoch]
    ax.scatter(best_epoch, best_value, color='red', zorder=10)
    ax.axvline(best_epoch, linestyle='--', color='red', alpha=0.5)
    ax.annotate(f'{best_value:.3f}',
                xy=(best_epoch, best_value),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot Metrics
    for i, metric in enumerate(metric_names, 1):
        ax = axes[i]
        train_metric = [e[metric] for e in train_metrics_list]
        val_metric = [e[metric] for e in val_metrics_list]

        #ax.plot(train_metric, alpha=0.3, color='tab:blue')
        #ax.plot(smooth(train_metric, smooth_factor), color='tab:blue',
        ax.plot(train_metric, color='tab:blue',
                label=f'Train {metric} (max: {np.max(train_metric):.3f})')

        #ax.plot(val_metric, alpha=0.3, color='tab:orange')
        #ax.plot(smooth(val_metric, smooth_factor), color='tab:orange',
        ax.plot(val_metric, color='tab:orange',
                label=f'Val {metric} (max: {np.max(val_metric):.3f})')

        if best_marker:
            best_epoch = np.argmax(val_metric)
            best_value = val_metric[best_epoch]
            ax.scatter(best_epoch, best_value, color='red', zorder=10)
            ax.axvline(best_epoch, linestyle='--', color='red', alpha=0.3)
            ax.annotate(f'{best_value:.3f}',
                        xy=(best_epoch, best_value),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        ax.set_ylabel(metric)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    # Final formatting
    axes[-1].set_xlabel('Epochs')
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()