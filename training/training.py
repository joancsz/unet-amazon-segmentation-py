
import torch
from torch.utils.tensorboard import SummaryWriter
from training.metrics import initialize_metrics, log_metrics, reset_metrics
from model.model import save_checkpoint

import time
from tqdm import tqdm
from copy import deepcopy

def run_epoch(model, loader, loss_fn, optimizer, scheduler, metrics, scaler, phase='train', device='cuda'):
    """Run one epoch of training/validation with proper gradient handling."""
    loss_total = 0.0
    pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch")

    # Phase-specific context managers
    torch.set_grad_enabled(phase == 'train')
    model.train() if phase == 'train' else model.eval()

    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=(phase == 'train')):  # Only enable AMP for training
            preds = model(imgs)
            loss = loss_fn(preds, masks.float())

        # Training-specific operations
        if phase == 'train':
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Unscale before clipping (CRUCIAL SAFETY STEP)
            scaler.unscale_(optimizer)

            # Gradient clipping (using unscaled gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Validation-specific handling
        else:
            # Ensure no gradient tracking in eval
            with torch.no_grad():  # Explicit safety even though grad is disabled
                preds_bin = (torch.sigmoid(preds) > 0.5).float()

        # Update metrics (safe for both phases)
        loss_total += loss.item()
        if phase == 'train':  # For training, preds_bin needs to be computed without no_grad
            preds_bin = (torch.sigmoid(preds) > 0.5).float()

        for metric in metrics.values():
            metric.update(preds_bin, masks.long())

        pbar.set_postfix({'Loss': loss.item()})

    # Final metric computation
    avg_loss = loss_total / len(loader)
    metrics_results = {name: metric.compute().item() for name, metric in metrics.items()}
    return avg_loss, metrics_results

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs=60, experiment_tag="", out_dir=None):
    """Train the model with early stopping, metric tracking, and checkpointing."""
    writer = SummaryWriter(comment=experiment_tag)
    device = next(model.parameters()).device  # Get device from model
    best_dice = 0.0
    best_model_state = None
    patience = 7
    no_improve = 0
    epoch_times = []
    epoch_val_metrics = []
    epoch_train_metrics = []

    # Initialize metrics for train/val
    metrics = {
        'train': initialize_metrics(device),
        'val': initialize_metrics(device)
    }
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss, train_metrics = run_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            metrics['train'], scaler, phase='train', device=device
        )

        # Validation Phase
        model.eval()
        val_loss, val_metrics = run_epoch(
            model, val_loader, loss_fn, None, None,
            metrics['val'], scaler, phase='val', device=device
        )

        # Logging & Checkpointing
        log_metrics(writer, epoch, train_loss, val_loss, train_metrics, val_metrics)
        epoch_val_metrics.append({'loss': val_loss, **val_metrics})
        epoch_train_metrics.append({'loss': train_loss, **train_metrics})

        # Early Stopping & Checkpoint
        if val_metrics['GeneralizedDice'] > best_dice:
            best_dice = val_metrics['GeneralizedDice']
            best_model_state = deepcopy(model.state_dict())
            save_checkpoint(model, experiment_tag, out_dir)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Scheduler Step
        scheduler.step(val_loss)
        epoch_times.append(time.time() - start_time)
        reset_metrics(metrics)

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    writer.close()

    return {
        "model": model,
        "best_dice": best_dice,
        "epoch_times": epoch_times,
        "total_time": sum(epoch_times),
        "avg_epoch_time": sum(epoch_times)/len(epoch_times),
        "epochs_run": epoch+1,
        "epoch_val_metrics": epoch_val_metrics,
        "epoch_train_metrics": epoch_train_metrics,
    }