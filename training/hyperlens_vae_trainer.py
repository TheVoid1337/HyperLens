import os
import torch
from dataclasses import dataclass
from torch.optim import Adam, AdamW
from tqdm import tqdm

from loss import DINOV2PerceptualLoss, KLDivergenceLoss, LatentConsistencyLoss
from model import HyperLensDCVAE, HyperLensDCVAEOutput
from torch.nn import MSELoss, L1Loss

from visualizer import VAETrainPlotter


@dataclass
class HyperLensVAEDCTrainerParams:
    model: HyperLensDCVAE
    batch_size: int = 128
    total_steps: int = 100_000
    save_every_steps: int = 5000
    log_every_steps: int = 1
    lr: float = 1e-4
    device: str = "cuda"
    optimizer: Adam | AdamW = AdamW
    recon_loss: MSELoss | L1Loss = L1Loss
    perceptual_loss: DINOV2PerceptualLoss = None
    kl_loss: KLDivergenceLoss = None
    beta_perceptual: float = 0.1
    gamma_kl: float = 1e-6
    save_path: str = "../checkpoints"
    model_filename: str = "hyperlens_vae_model.pth"
    image_save_path: str = "images/"
    phase_training: str = "phase_1" # "phase_1" or "phase_2"
    save_recons_every_steps = 10000
    lambda_consistency: float = 0.01


class HyperLensVAEDCTrainer:
    def __init__(self, params: HyperLensVAEDCTrainerParams):
        self.params = params
        self.device = params.device
        self.model = params.model.to(params.device)
        self.recon_loss = params.recon_loss(reduction='mean')
        self.perceptual_loss = params.perceptual_loss
        self.kl_loss = params.kl_loss
        self.total_steps = params.total_steps
        self.save_every_steps = params.save_every_steps
        self.log_every_steps = params.log_every_steps

        self.save_path = params.save_path
        self.model_filename = params.model_filename
        self.visualizer = None
        self.optimizer = None

        self.latent_loss_fn = params.kl_loss() if params.kl_loss else None
        self.perceptual_loss_fn = params.perceptual_loss(image_size=224) if params.perceptual_loss else None
        self.scaler = torch.amp.GradScaler()
        self.consistency_loss_fn = LatentConsistencyLoss()
        self.lambda_consistency = params.lambda_consistency

        self.phase_training = params.phase_training
        self.save_recons_every_steps = params.save_recons_every_steps
        self.compiled_model = None
        self.print_model_summary()

    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)

        print("\n" + "=" * 50)
        print("[i] MODEL SUMMARY")
        print("=" * 50)
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Model Size (Weights):  {total_size_mb:.2f} MB")
        print("=" * 50 + "\n")


    def setup_training_phase(self):
        phase = self.phase_training.lower()
        print(f"\n[🚀] INITIALIZING TRAINING: {phase.upper()} [🚀]")

        for param in self.model.parameters():
            param.requires_grad = False

        active_params_names = []

        if phase == "phase_1" or phase == "phase_1_2":
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                active_params_names.append(name)
            print("[*] All layers are active!")

        elif phase == "phase_2":
            # encoder_layers = ['encoder.output_norm', 'encoder.output_conv']
            # for name, param in self.model.named_parameters():
            #     if any(layer in name for layer in encoder_layers):
            #         param.requires_grad = True
            #         active_params_names.append(name)
            #
            # for idx in range(len(self.model.encoder.encoder) - 2, len(self.model.encoder.encoder)):
            #     for name, param in self.model.encoder.encoder[idx].named_parameters():
            #         param.requires_grad = True
            #         active_params_names.append(f"encoder.encoder.{idx}.{name}")

            for name, param in self.model.decoder.named_parameters():
                param.requires_grad = True
                active_params_names.append(name)

        elif phase == "phase_3":
            phase3_layers = ['decoder.norm_out', 'decoder.output_conv']
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in phase3_layers):
                    param.requires_grad = True
                    active_params_names.append(name)

            print("[*] Only decoder output active!")
        else:
            raise ValueError(f"Unknown phase: {phase}")

        active_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.params.optimizer(active_params, lr=self.params.lr)
        print(f"[*] VAE-Optimizer created with {len(active_params)} active tensors.")

    def save_model(self, path: str, step: int, loss: float):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, path)
        print(f"Model and states successfully saved to {path}")


    def load_model(self, path: str):
        if os.path.exists(path):
            print(f"Loading checkpoint from {path}...")
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            step = checkpoint.get('step', 0)
            loss = checkpoint.get('loss', 0.0)
            print(f"Model successfully loaded! (Resuming from Step {step}, Loss: {loss:.4f})")
        else:
            print(f"No checkpoint found at {path}. Starting from scratch.")
            step = 0
            loss = float('inf')

        print("Compiling model...")
        self.compiled_model = torch.compile(model=self.model, mode="default")
        self.visualizer = VAETrainPlotter(self.compiled_model, self.device)
        return step, loss

    def calculate_loss(self, real_images, model_output: HyperLensDCVAEOutput):
        recon_loss = self.recon_loss(model_output.reconstructed_image, real_images)
        kl_loss = torch.tensor(0.0, device=self.device)
        perceptual_loss = torch.tensor(0.0, device=self.device)

        if self.latent_loss_fn and not self.phase_training == "phase_3":
            kl_loss = self.latent_loss_fn(model_output.tangent_means, model_output.tangent_log_vars)

        if self.perceptual_loss_fn:
            perceptual_loss = self.perceptual_loss_fn(model_output.reconstructed_image, real_images)

        return recon_loss, kl_loss, perceptual_loss

    def calculate_consistency_loss(self, real_means, real_log_vars, fake_means, fake_log_vars):

        consistency_loss = self.consistency_loss_fn(real_means, real_log_vars, fake_means, fake_log_vars)
        return consistency_loss


    def train(self, dataloader):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        if not os.path.exists(self.params.image_save_path):
            os.makedirs(self.params.image_save_path, exist_ok=True)

        avg_loss, log_loss, log_recon, log_kl, log_percept, log_consistency = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        consistency_loss = torch.tensor(0.0, device=self.device)

        start_step, best_loss = self.load_model(os.path.join(self.save_path, self.model_filename))
        self.setup_training_phase()

        print(f"Start Training from Step {start_step} to {self.total_steps}...")

        data_iter = iter(dataloader)
        progress_bar = tqdm(range(start_step, self.total_steps), desc="Training Steps")

        for step in progress_bar:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                images, labels = next(data_iter)

            images = images.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                output = self.compiled_model(images)


                recon_loss, kl_loss, perceptual_loss = self.calculate_loss(images, output)

                if self.phase_training == "phase_1_2" and step >= self.params.total_steps - 10_000:

                    encoder_params = self.model.encoder.parameters()
                    for param in encoder_params:
                        param.requires_grad = False

                    real_means, real_log_vars = output.tangent_means, output.tangent_log_vars
                    fake_encoded = self.model.encode(output.reconstructed_image)
                    fake_means, fake_log_vars = fake_encoded.tangent_means, fake_encoded.tangent_log_vars
                    consistency_loss = self.calculate_consistency_loss(real_means, real_log_vars,
                                                                       fake_means, fake_log_vars)

                    for param in encoder_params:
                        param.requires_grad = True


                loss = (recon_loss + self.params.beta_perceptual * perceptual_loss
                        + self.params.gamma_kl * kl_loss + self.lambda_consistency * consistency_loss)


            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            self.scaler.update()

            log_loss += loss.item()
            log_recon += recon_loss.item()
            log_kl += kl_loss.item()
            log_percept += perceptual_loss.item()
            log_consistency += consistency_loss.item()

            if (step + 1) % self.log_every_steps == 0:
                avg_loss = log_loss / self.log_every_steps
                progress_bar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'Recon': f"{log_recon / self.log_every_steps:.4f}",
                    'DINO': f"{log_percept / self.log_every_steps:.4f}",
                    'KL': f"{log_kl / self.log_every_steps:.4f}",
                    'Consistency': f"{log_consistency / self.log_every_steps}"
                })
                log_loss, log_recon, log_kl, log_percept, log_consistency = 0, 0, 0, 0, 0

            if (step + 1) % self.save_every_steps == 0:
                print(f"\n[Step {step + 1}] Saving Checkpoint...")
                save_dir = os.path.join(self.save_path, f"{self.model_filename}")
                self.save_model(save_dir, step + 1, avg_loss if 'avg_loss' in locals() else loss.item())

            if (step + 1) % self.save_recons_every_steps == 0:
                self.model.eval()
                plot_path = f"{self.params.image_save_path}/reconstruction_step_{step + 1}.jpeg"
                self.visualizer.plot_reconstructions(dataloader, num_images=8, save_path=plot_path)
                self.model.train()





