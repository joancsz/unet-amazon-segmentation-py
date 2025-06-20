import segmentation_models_pytorch as smp
import torch


def get_model(encoder, decoder_attention_type, device, in_channels=4, classes=1):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
        decoder_attention_type=decoder_attention_type
    ).to(device)

def save_checkpoint(model, tag, out_dir):
    """Save model checkpoint."""
    path = f"{out_dir}/best_model_{tag}.pth" if out_dir else f"best_model_{tag}.pth"
    torch.save(model.state_dict(), path)