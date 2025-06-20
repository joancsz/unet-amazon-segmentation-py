import torch
import torch.nn as nn
import torch.optim as optim

from thop import profile, clever_format

from training.training import train_model
from model.model import get_model
from model.loss import DiceBCELoss

def run_experiment(config, train_loader, val_loader, epochs=50):
    """Run a full experiment with model initialization, training, and evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_tag = f"{config['encoder']}_{config.get('decoder_attention', 'none')}"

    # Model Initialization
    model = get_model(config['encoder'], config.get('decoder_attention', None), device)

    # Compute Model Complexity
    flops, params = "N/A", "N/A"
    try:
        input = torch.randn(1, 4, 512, 512).to(device)
        flops, params = profile(model, inputs=(input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
    except Exception as e:
        print(f"Error computing FLOPs: {e}")

    print(f"\nExperiment: {experiment_tag}")
    print(f"Params: {params}, FLOPs: {flops}")

    # Training Setup
    loss_fn = DiceBCELoss(weight_bce=0.5, weight_dice=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Track GPU Memory
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Train Model
    metrics = train_model(model, train_loader, val_loader, loss_fn, optimizer,
                         scheduler, epochs=epochs, experiment_tag=experiment_tag,
                         out_dir=config['out_dir'])

    # Finalize Metrics
    metrics.update({
        "peak_memory_MB": torch.cuda.max_memory_allocated(device) / (1024**2) if device == 'cuda' else 'N/A',
        "flops": flops,
        "params": params,
        "config": config,
        "experiment_tag": experiment_tag
    })
    return metrics