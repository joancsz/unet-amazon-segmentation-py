from torch.utils.data import DataLoader
import torch
import os
import pickle

from config.config import second_config as cfg                     

from training.experiment import run_experiment
from model.model import get_model
from data.dataset import SatelliteSegmentationDataset
from data.transform import get_transforms
from evaluation.evaluation import visualize_predictions_v2, plot_training_curves, evaluate_model


transform = get_transforms()
train_dataset = SatelliteSegmentationDataset(cfg.train_img_dir, cfg.train_mask_dir, transform)
val_dataset = SatelliteSegmentationDataset(cfg.val_img_dir, cfg.val_mask_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

os.makedirs(cfg.model_runs_output, exist_ok=True)
experiment_results = []

for config in cfg.experiment_configs:
    os.makedirs(config['out_dir'], exist_ok=True)
    result = run_experiment(config, train_loader, val_loader, epochs=1)
    experiment_results.append(result)


with open(f'{cfg.model_runs_output}/experiment_results.pkl', 'wb') as f:
  pickle.dump(experiment_results, f)

#Evaluate
for result, config in zip(experiment_results, cfg.experiment_configs):
    PATH = config['out_dir']
    model_path = os.path.join(PATH, f"best_model_{config['encoder']}_None.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config['encoder'], config.get('decoder_attention', None), device)
    state_dict = torch.load(model_path)
    filtered_state_dict = {k: v for k, v in state_dict.items()
                        if not k.endswith('total_ops') and not k.endswith('total_params')}

    model.load_state_dict(filtered_state_dict, strict=False)
    for seed in [42, 37, 21]:
        visualize_predictions_v2(model, val_dataset, num_samples=3, random_seed=seed, save_path=os.path.join(PATH, f"predictions_{seed}_{config['encoder']}.png"))
    evaluate_model(model, val_loader, save_path=PATH)

    plot_training_curves(result['epoch_train_metrics'],
                    result['epoch_val_metrics'],
                    title=f"Training Curves ({config['encoder']})",
                    save_path=f"{PATH}/training_curves.png")