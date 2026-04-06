import json

from torch.nn import L1Loss

from loss import KLDivergenceLoss, DINOV2PerceptualLoss
from model import ManifoldComponent, generate_stage_configurations, HyperLensVAEModelParams, HyperLensDCVAE
from training import HyperLensVAEDCTrainer, HyperLensVAEDCTrainerParams
from dataloader import InfiniteImageNetLoader

def load_config(config_path="config/vae_train_params.json"):
    with open(config_path, "r") as f:
        return json.load(f)






if __name__ == '__main__':
    config = load_config()

    components = [
        ManifoldComponent(manifold_type='euclidean', dimension=64),
        ManifoldComponent(manifold_type='hyperbolic', dimension=128, curvature=1.0),
        ManifoldComponent(manifold_type='spherical', dimension=64)
    ]

    # TODO add to model
    channels_list, depth_list = generate_stage_configurations(**config["stage_configurations"])

    model_params = HyperLensVAEModelParams(
        manifold_components=[ManifoldComponent(**comp_kwargs) for comp_kwargs in config["manifolds"]],
        channel_list=channels_list,
        depth_list=depth_list,
        **config["model"]
    )

    model = HyperLensDCVAE(model_params)

    trainer_kwargs = config["trainer"]

    trainer_kwargs["recon_loss"] = L1Loss
    trainer_kwargs["kl_loss"] = KLDivergenceLoss
    trainer_kwargs["perceptual_loss"] = DINOV2PerceptualLoss

    dataset_config = config["dataset"]
    ssd_cache_dir = dataset_config.pop("ssd_cache_dir")

    params = HyperLensVAEDCTrainerParams(
        model=model,
        **trainer_kwargs
    )

    trainer = HyperLensVAEDCTrainer(params)

    dataloader = InfiniteImageNetLoader(
        ssd_cache_dir,
        **dataset_config
    )

    trainer.train(dataloader)