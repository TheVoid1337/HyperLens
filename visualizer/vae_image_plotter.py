import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class VAETrainPlotter:
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    def plot_reconstructions(self, dataloader, num_images: int = 8, save_path: str = None):

        with torch.no_grad():
            batch = next(iter(dataloader))
            images, _ = batch

            images = images[:num_images].to(self.device)

            output = self.model(images)

            reconstructed = output.reconstructed_image

            reconstructed = (reconstructed + 1.0) / 2.0

            reconstructed = torch.clamp(reconstructed, 0.0, 1.0)

            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)


            real_grid = vutils.make_grid(images.cpu(), nrow=num_images, padding=2, normalize=False)
            recon_grid = vutils.make_grid(reconstructed.cpu(), nrow=num_images, padding=2, normalize=False)

            fig, axes = plt.subplots(2, 1, figsize=(num_images * 2, 4))

            axes[0].imshow(real_grid.permute(1, 2, 0).numpy())
            axes[0].set_title("Original", fontsize=14)
            axes[0].axis("off")

            axes[1].imshow(recon_grid.permute(1, 2, 0).numpy())
            axes[1].set_title("VAE Reconstruction", fontsize=14)
            axes[1].axis("off")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plots saved at: {save_path}")
            else:
                plt.show()

            plt.close()