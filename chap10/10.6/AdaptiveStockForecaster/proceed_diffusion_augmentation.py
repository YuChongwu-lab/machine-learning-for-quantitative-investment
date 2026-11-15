"""
Proceed + Diffusion data-augmentation integration module.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import random


# ═══════════════════════════════════════════════════════════════════
#  Diffusion model
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeSeriesDenoiser(nn.Module):
    """
    Time-series denoising network.
    """
    def __init__(self, seq_len, feature_dim, hidden_dim, context_dim, num_steps):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_steps),
            nn.Linear(num_steps, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Temporal convolution
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Context projection layer
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm4 = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x_t, t, context):
        """
        Args:
            x_t: (B, T, F) noisy data
            t: (B,) timestep
            context: (B, C) context vector
        Returns:
            noise: (B, T, F) predicted noise
        """
        # Timestep embedding
        t_emb = self.time_mlp(t)  # (B, H)

        # Input projection
        h = self.input_proj(x_t)  # (B, T, H)

        # Temporal convolution
        h_conv = h.transpose(1, 2)  # (B, H, T)
        h_conv = F.gelu(self.conv1(h_conv))
        h_conv = h_conv + t_emb.unsqueeze(-1)
        h_conv = F.gelu(self.conv2(h_conv))
        h = h_conv.transpose(1, 2)
        h = self.norm1(h)

        # Self-attention
        h_attn, _ = self.self_attn(h, h, h)
        h = h + h_attn
        h = self.norm2(h)

        # Context cross-attention
        context_expanded = self.context_proj(context).unsqueeze(1)  # (B, 1, H)
        h_cross, _ = self.cross_attn(h, context_expanded, context_expanded)
        h = h + h_cross
        h = self.norm3(h)

        # Output
        noise = self.output_proj(h)
        return noise


class SimpleDDPM(nn.Module):
    """
    A simplified diffusion model (with a corrected DDIM sampler).
    """

    def __init__(self,
                 seq_len: int = 24,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 context_dim: int = 64,
                 num_steps: int = 50,
                 sampling_steps: int = 10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.sampling_steps = sampling_steps

        # Noise schedule
        self.register_noise_schedule()

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )

        # Denoising network
        self.denoiser = TimeSeriesDenoiser(
            seq_len=seq_len,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_steps=num_steps
        )

    def register_noise_schedule(self, beta_start=1e-4, beta_end=0.02):
        """Register the noise schedule (with added numerical stability)."""
        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Numerical-stability clamp
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-8)))

    def forward_diffusion(self, x_0, t, noise=None):
        """Forward noising process."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, context, n_samples=1, eta=0.0):
        """
        Args:
            context: context vector
            n_samples: number of samples to generate per context
            eta: randomness parameter (0 = deterministic, 1 = DDPM)
        """
        B = context.size(0)
        device = context.device

        # Expand the context
        context = context.repeat_interleave(n_samples, dim=0)

        # Start from pure noise
        x_t = torch.randn(B * n_samples, self.seq_len, self.feature_dim, device=device)

        # Choose the sampling timesteps
        timesteps = torch.linspace(
            self.num_steps - 1, 0, self.sampling_steps, dtype=torch.long, device=device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B * n_samples,), t, device=device, dtype=torch.long)

            # Predict the noise
            predicted_noise = self.denoiser(x_t, t_batch, context)

            # Get the alpha values
            alpha_t = self.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # The corrected DDIM update formula
            # predict x_0 (with added numerical stability)
            sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))

            x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

            if i < len(timesteps) - 1:
                # Compute the variance (optional stochasticity)
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
                )

                # Directional noise term
                direction = torch.sqrt(torch.clamp(
                    1 - alpha_t_prev - sigma_t ** 2, min=0
                )) * predicted_noise

                # Random noise (only if eta > 0)
                noise = torch.randn_like(x_t) if eta > 0 else 0

                # Update
                x_t = torch.sqrt(alpha_t_prev) * x_0_pred + direction + sigma_t * noise
            else:
                x_t = x_0_pred

        return x_t

    def training_loss(self, x_0, context):
        """Training loss."""
        B = x_0.size(0)
        device = x_0.device

        # Random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # Forward noising
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_diffusion(x_0, t, noise)

        # Predict the noise
        predicted_noise = self.denoiser(x_t, t, context)

        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss


# ═══════════════════════════════════════════════════════════════════
# Diffusion augmentor
# ═══════════════════════════════════════════════════════════════════

class DiffusionAugmentor:
    """
    Diffusion-based data augmentor.
    """

    def __init__(self, model: SimpleDDPM, mode='faithful', adversarial_strength=0.3):
        """
        Args:
            model: a SimpleDDPM model
            mode: augmentation mode
                - 'faithful': faithfully reproduces the original data distribution
                - 'adversarial': generates adversarial samples
                - 'mixed': a mix of both modes
            adversarial_strength: adversarial strength (0-1)
        """
        self.model = model
        self.mode = mode
        self.adversarial_strength = adversarial_strength
        self.device = next(model.parameters()).device

    def pretrain(self, X_train, epochs=10, batch_size=32, lr=1e-3, save_path=None):
        """
        Pretrain the diffusion model.
        Args:
            X_train: (N, T, F) training data
            epochs: number of training epochs
            batch_size: batch size
            lr: learning rate
            save_path: path to save the model
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"Starting pretraining (epochs={epochs}, batch_size={batch_size}, samples={n_samples})")

        for epoch in range(epochs):
            epoch_loss = 0.0
            # shuffle
            indices = torch.randperm(n_samples)

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # fetch the batch
                x_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)

                # encode the context
                B, T, F = x_batch.shape
                with torch.no_grad():
                    context = self.model.context_encoder(x_batch.view(B, -1))

                # training loss
                loss = self.model.training_loss(x_batch, context)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # save the model
        if save_path:
            self.save_pretrained(save_path)
            print(f"✓ Model saved: {save_path}")

        self.model.eval()

    def save_pretrained(self, save_path):
        """Save the pretrained model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.model.seq_len,
            'feature_dim': self.model.feature_dim,
            'num_steps': self.model.num_steps,
            'sampling_steps': self.model.sampling_steps,
            'mode': self.mode,
            'adversarial_strength': self.adversarial_strength
        }
        torch.save(checkpoint, save_path)

    def load_pretrained(self, load_path):
        """
        Load a pretrained model (automatically handles a config mismatch).

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(load_path):
            print(f"⚠️ File does not exist: {load_path}")
            return False

        try:
            print(f"Attempting to load model: {load_path}")
            ckpt = torch.load(load_path, map_location=self.device, weights_only=False)

            # Check whether the config matches
            ckpt_feature_dim = ckpt.get('feature_dim', None)
            ckpt_seq_len = ckpt.get('seq_len', None)

            config_mismatch = False
            mismatch_reasons = []

            if ckpt_feature_dim != self.model.feature_dim:
                mismatch_reasons.append(
                    f"feature_dim: expected {self.model.feature_dim}, got {ckpt_feature_dim}"
                )
                config_mismatch = True

            if ckpt_seq_len != self.model.seq_len:
                mismatch_reasons.append(
                    f"seq_len: expected {self.model.seq_len}, got {ckpt_seq_len}"
                )
                config_mismatch = True

            if config_mismatch:
                print(f"⚠️ Config mismatch:")
                for reason in mismatch_reasons:
                    print(f"  - {reason}")

                # Remove the mismatched checkpoint
                print(f"  -> removing the mismatched checkpoint")
                try:
                    os.remove(load_path)
                    print(f"  ✓ removed; retraining is required")
                except Exception as e:
                    print(f"  ✗ failed to remove: {e}")

                return False  # False means retraining is needed

            # Config matches; load the model
            self.model.load_state_dict(ckpt['model_state_dict'])

            # Restore the remaining config
            if 'mode' in ckpt:
                self.mode = ckpt['mode']
            if 'adversarial_strength' in ckpt:
                self.adversarial_strength = ckpt['adversarial_strength']

            print(f"✓ Model loaded successfully (feature_dim={self.model.feature_dim}, seq_len={self.model.seq_len})")
            self.model.eval()
            return True

        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            print(f"  -> attempting to remove the corrupted checkpoint")
            try:
                os.remove(load_path)
                print(f"  ✓ removed the corrupted checkpoint")
            except:
                pass
            return False

    @torch.no_grad()
    def augment(self, X_batch, n_augment=None):
        """
        Generate augmented samples.

        Args:
            X_batch: (B, T, F) real samples
            n_augment: number of augmented samples to generate per sample (default = B)

        Returns:
            X_aug: (B*n, T, F) augmented samples
        """
        self.model.eval()
        B, T, F = X_batch.shape
        device = X_batch.device

        if n_augment is None:
            n_augment = B

        # Encode the context
        context = self.model.context_encoder(X_batch.view(B, -1))

        # Generate samples according to the mode
        if self.mode == 'faithful':
            # faithful mode: low randomness
            X_aug = self.model.ddim_sample(context, n_samples=n_augment, eta=0.0)

        elif self.mode == 'adversarial':
            # adversarial mode: perturb the context
            noise_scale = self.adversarial_strength
            context_perturbed = context + torch.randn_like(context) * noise_scale
            X_aug = self.model.ddim_sample(context_perturbed, n_samples=n_augment, eta=0.5)

        elif self.mode == 'mixed':
            # mixed mode: partly faithful, partly adversarial
            n_faithful = n_augment // 2
            n_adversarial = n_augment - n_faithful

            X_faithful = self.model.ddim_sample(context, n_samples=n_faithful, eta=0.0)

            noise_scale = self.adversarial_strength
            context_perturbed = context + torch.randn_like(context) * noise_scale
            X_adversarial = self.model.ddim_sample(context_perturbed, n_samples=n_adversarial, eta=0.5)

            X_aug = torch.cat([X_faithful, X_adversarial], dim=0)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return X_aug

    def evaluate_quality(self, X_real, n_samples=100):
        """
        Evaluate the quality of generated samples.
        Args:
            X_real: (N, T, F) real samples
            n_samples: number of generated samples
        Returns:
            metrics: a dict of quality metrics
        """
        self.model.eval()

        # randomly select samples
        n_real = min(n_samples, len(X_real))
        indices = np.random.choice(len(X_real), n_real, replace=False)
        X_batch = torch.FloatTensor(X_real[indices]).to(self.device)

        # generate samples
        with torch.no_grad():
            X_gen = self.augment(X_batch, n_augment=1)

        X_real_np = X_batch.cpu().numpy()
        X_gen_np = X_gen.cpu().numpy()

        # compute statistics
        metrics = {}

        # KS statistic
        from scipy import stats
        ks_stats = []
        for f in range(X_real_np.shape[-1]):
            ks_stat, _ = stats.ks_2samp(
                X_real_np[:, :, f].flatten(),
                X_gen_np[:, :, f].flatten()
            )
            ks_stats.append(ks_stat)
        metrics['ks_statistic'] = np.mean(ks_stats)

        # mean and std difference
        metrics['mean_diff'] = np.abs(X_real_np.mean() - X_gen_np.mean())
        metrics['std_diff'] = np.abs(X_real_np.std() - X_gen_np.std())

        # autocorrelation (optional)
        try:
            real_autocorr = np.mean([
                np.corrcoef(X_real_np[i, :-1, 0], X_real_np[i, 1:, 0])[0, 1]
                for i in range(min(100, len(X_real_np)))
            ])
            gen_autocorr = np.mean([
                np.corrcoef(X_gen_np[i, :-1, 0], X_gen_np[i, 1:, 0])[0, 1]
                for i in range(min(100, len(X_gen_np)))
            ])
            metrics['autocorr_diff'] = np.abs(real_autocorr - gen_autocorr)
        except:
            pass

        return metrics


# ═══════════════════════════════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════════════════════════════

class ProceedDiffusionDataset(Dataset):
    """
    Proceed + Diffusion dataset.
    """

    def __init__(self,
                 Xw_all,
                 Yw_all,
                 diffusion_augmentor: Optional[DiffusionAugmentor] = None,
                 batch_size=32,
                 augment_prob=0.35,
                 augment_ratio=1.0,
                 use_concept_only=True):
        """
        Args:
            Xw_all: (N, T, F) windowed data
            Yw_all: (N, P) labels
            diffusion_augmentor: a diffusion augmentor (optional)
            batch_size: batch size
            augment_prob: probability of applying augmentation
            augment_ratio: ratio of generated data (relative to real data)
            use_concept_only: whether augmentation is only used during concept encoding
        """
        self.Xw_all = Xw_all
        self.Yw_all = Yw_all
        self.augmentor = diffusion_augmentor
        self.batch_size = batch_size
        self.augment_prob = augment_prob
        self.augment_ratio = augment_ratio
        self.use_concept_only = use_concept_only

        self.n_total = len(Xw_all)
        self.n_batches = (self.n_total + batch_size - 1) // batch_size

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_total)

        Xb = self.Xw_all[start_idx:end_idx]
        Yb = self.Yw_all[start_idx:end_idx]

        # whether to apply augmentation
        use_augment = (
                self.augmentor is not None and
                random.random() < self.augment_prob
        )

        if use_augment:
            try:
                n_real = len(Xb)
                n_gen = max(1, int(n_real * self.augment_ratio))

                # generate augmented samples
                Xb_tensor = torch.FloatTensor(Xb).to(next(self.augmentor.model.parameters()).device)
                Xb_aug = self.augmentor.augment(Xb_tensor, n_augment=n_gen).cpu().numpy()

                if self.use_concept_only:
                    # augmented data is only used during concept encoding
                    # return the mixed data + a marker of how many real samples it has
                    Xb_mixed = np.concatenate([Xb, Xb_aug], axis=0)
                    return torch.FloatTensor(Xb_mixed), torch.FloatTensor(Yb), n_real
                else:
                    # augmented data also participates in training
                    Yb_aug = Yb  # duplicate the labels (simplified handling)
                    Xb_mixed = np.concatenate([Xb, Xb_aug], axis=0)
                    Yb_mixed = np.concatenate([Yb, Yb_aug], axis=0)
                    return torch.FloatTensor(Xb_mixed), torch.FloatTensor(Yb_mixed)

            except Exception as e:
                print(f"⚠️ Augmentation failed: {e}, falling back to the original data")
                return torch.FloatTensor(Xb), torch.FloatTensor(Yb)

        return torch.FloatTensor(Xb), torch.FloatTensor(Yb)


# ═══════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════

def setup_diffusion_augmentor(Xw_train, seq_len, feature_dim,
                              mode='mixed', device='cuda',
                              pretrain_epochs=5, save_path=None):
    """
    Convenience function: initialize and pretrain a diffusion augmentor.

    Whether to train is decided by load_pretrained's return value.
    """
    print("\n" + "=" * 70)
    print("Initializing the diffusion data augmentor")
    print("=" * 70)

    # Build the model
    diffusion_model = SimpleDDPM(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=64,
        context_dim=64,
        num_steps=50,
        sampling_steps=10
    ).to(device)

    # Build the augmentor
    augmentor = DiffusionAugmentor(
        diffusion_model,
        mode=mode,
        adversarial_strength=0.3
    )

    # Pretrain or load
    need_training = True

    if save_path and os.path.exists(save_path):
        print(f"Found an existing model: {save_path}")
        loaded = augmentor.load_pretrained(save_path)

        if loaded:
            print(f" Successfully loaded the existing model")
            need_training = False
        else:
            print(f" Failed to load or config mismatch; retraining")
            need_training = True

    if need_training:
        print("Pretraining the diffusion model...")
        augmentor.pretrain(
            Xw_train,
            epochs=pretrain_epochs,
            batch_size=32,
            lr=1e-3,
            save_path=save_path
        )

    # Evaluate quality
    print("\nEvaluating generation quality...")
    metrics = augmentor.evaluate_quality(Xw_train[:100])
    if metrics:
        print(f"KS statistic: {metrics.get('ks_statistic', 0):.4f}")
        print(f"Mean difference: {metrics.get('mean_diff', 0):.4f}")
        print(f"Std difference: {metrics.get('std_diff', 0):.4f}")
        if 'autocorr_diff' in metrics:
            print(f"Autocorrelation difference: {metrics['autocorr_diff']:.4f}")

    print(" Diffusion augmentor is ready!")
    print("=" * 70 + "\n")

    return augmentor

