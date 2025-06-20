import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
from pathlib import Path

class BaseExperimentConfig(BaseSettings):
    """Base configuration for all experiments"""
    data_dir: Path
    model_runs_output: Path
    train_img_dir: Path
    train_mask_dir: Path
    val_img_dir: Path
    val_mask_dir: Path
    test_img_dir: Path
    test_mask_dir: Path
    experiment_configs: List[Dict[str, Optional[str]]]

    class Config:
        # Use environment variables to override these values
        env_prefix = "experiment_"
        case_sensitive = False

    def __init__(self, **data):
        # Convert string paths to Path objects
        for field, value in data.items():
            #Will ignore experiment_configs
            if isinstance(value, str) and any(field.endswith(x) for x in ["dir", "output"]):
                data[field] = Path(value)
        super().__init__(**data)


class FirstExperimentConfig(BaseExperimentConfig):
    """Configuration for the first experiment"""
    def __init__(self):
        data_dir = Path("/AMAZON")
        model_runs_output = Path("/results")
        
        super().__init__(
            data_dir=data_dir,
            model_runs_output=model_runs_output,
            train_img_dir=data_dir / "Training/images",
            train_mask_dir=data_dir / "Training/label",
            val_img_dir=data_dir / "Test/images",
            val_mask_dir=data_dir / "Test/label",
            test_img_dir=data_dir / "Validation/images",
            test_mask_dir=data_dir / "Validation/label",
            experiment_configs=[
                {"encoder": "resnet18", "decoder_attention": None, "out_dir": model_runs_output / "resnet18"},
                {"encoder": "resnet34", "decoder_attention": None, "out_dir": model_runs_output / "resnet34"},
                {"encoder": "efficientnet-b0", "decoder_attention": None, "out_dir": model_runs_output / "efficientnet-b0"},
                {"encoder": "mobilenet_v2", "decoder_attention": None, "out_dir": model_runs_output / "mobilenet_v2"}
            ]
        )


class SecondExperimentConfig(BaseExperimentConfig):
    """Configuration for the second experiment"""
    def __init__(self):
        data_dir = Path("/CUSTOM_AMAZON")
        model_runs_output = Path("results_custom")
        
        super().__init__(
            data_dir=data_dir,
            model_runs_output=model_runs_output,
            train_img_dir=data_dir / "Training/images",
            train_mask_dir=data_dir / "Training/labels",
            val_img_dir=data_dir / "Test/images",
            val_mask_dir=data_dir / "Test/labels",
            test_img_dir=data_dir / "Validation/images",
            test_mask_dir=data_dir / "Validation/labels",
            experiment_configs=[
                {"encoder": "resnet18", "decoder_attention": None, "out_dir": model_runs_output / "resnet18"},
            ]
        )


# Initialize configurations
first_config = FirstExperimentConfig()
second_config = SecondExperimentConfig()