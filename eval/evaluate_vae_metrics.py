import json
import torch
from torch import Tensor
import torch.nn.functional as fun
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

from model import ManifoldComponent, HyperLensDCVAE, HyperLensVAEModelParams
from model import generate_stage_configurations


@torch.no_grad()
def evaluate_metrics_combined(model, dataloader, device, recon_attr="reconstructed_image"):
    print(f"[*] Starting evaluation on device: {device.upper()}...")
    model.eval()

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr = 0.0
    total_samples = 0
    eps = 1e-8

    progress_bar = tqdm(dataloader, desc="[🔍] Evaluation on validation images")

    for batch in progress_bar:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            output = model(images)
            recons = getattr(output, recon_attr) if hasattr(output, recon_attr) else output

        images = to_01(images).float()
        recons = to_01(recons).float()

        fid_metric.update(images, real=True)
        fid_metric.update(recons, real=False)

        ssim_metric.update(recons, images)

        mse = fun.mse_loss(recons, images, reduction="none")
        mse = mse.flatten(1).mean(dim=1)
        psnr = 10.0 * torch.log10(1.0 / (mse + eps))

        total_psnr += psnr.sum().item()
        total_samples += psnr.numel()

    print("[*] Computing finale Scores")
    final_fid = float(fid_metric.compute().item())
    final_ssim = float(ssim_metric.compute().item())
    final_psnr = total_psnr / total_samples

    return final_fid, final_ssim, final_psnr


def load_config(config_path="config/vae_train_params.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def to_01(x: Tensor) -> Tensor:
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1.0) / 2.0
    else:
        x = x.clamp(0, 1)
    return x


def transform_val_batch(examples):
    examples["pixel_values"] = [val_transform(img.convert("RGB")) for img in examples["image"]]
    return examples


def custom_collate(batch):

    images = torch.stack([item["pixel_values"] for item in batch])
    return images


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config("../config/vae_eval_params.json")
    print("[*] Loading ImageNet validation subset...")

    val_dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        cache_dir=config["dataset"]["ssd_cache_dir"]
    )

    image_size = config["dataset"]["image_size"]

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset.set_transform(transform_val_batch)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        collate_fn=custom_collate
    )

    print("[*] Loading Model-Checkpoint...")

    channels_list, depth_list = generate_stage_configurations(**config["stage_configurations"])

    model_params = HyperLensVAEModelParams(
        manifold_components=[ManifoldComponent(**comp_kwargs) for comp_kwargs in config["manifolds"]],
        channel_list=channels_list,
        depth_list=depth_list,
        **config["model"]
    )
    model = HyperLensDCVAE(model_params).to(device)

    model.load_state_dict(torch.load("../checkpoints/vae/hyperlens_vae_model.pth", map_location=device)["model_state_dict"])

    fid, ssim, psnr = evaluate_metrics_combined(model, val_loader, device)

    print("\n" + "=" * 40)
    print("VALIDATION RESULTS (Unseen Images)")
    print("=" * 40)
    print(f"FID Score:  {fid:.2f}")
    print(f"SSIM Score: {ssim:.2f}")
    print(f"PSNR Score: {psnr:.2f} dB")
    print("=" * 40)
