import torch
import torch.nn as nn
import torchvision.transforms as transform
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

class DINOV2PerceptualLoss(nn.Module):
    def __init__(self, device="cuda", model_name='dinov2_vits14', image_size=224):
        super(DINOV2PerceptualLoss, self).__init__()
        self.device = device

        print(f"Loading {model_name} as Perceptual Loss...")

        self.dino = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.dino.eval()


        for param in self.dino.parameters():
            param.requires_grad = False

        self.transform = transform.Compose([
            transform.Resize((image_size, image_size), antialias=True),
            #  transform.Lambda(lambda x: (x + 1.0) / 2.0), use if necessary
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.feature_loss = nn.MSELoss()

        self.blocks_to_take = [2, 5, 8, 11]

    def forward(self, recon_images, real_images):

        recon_dino = self.transform(recon_images)
        real_dino = self.transform(real_images)

        with torch.no_grad():
            real_features = self.dino.get_intermediate_layers(real_dino, n=self.blocks_to_take)

        recon_features = self.dino.get_intermediate_layers(recon_dino, n=self.blocks_to_take)

        total_loss = 0.0
        for real_feat, recon_feat in zip(real_features, recon_features):
            total_loss += self.feature_loss(recon_feat, real_feat)

        return total_loss

