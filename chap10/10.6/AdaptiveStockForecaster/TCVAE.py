import torch
import torch.nn as nn
import torch.nn.functional as F

class TCVAE(nn.Module):
    """
    Temporal Conditional Variational Autoencoder (TCVAE).
    Architecture:
        - Encoder: Conv1d layers with stride=2 for temporal downsampling
        - Latent layer: Gaussian distribution (mu, logvar)
        - Decoder: ConvTranspose1d layers for temporal upsampling
    Args:
        feature_dim: input feature dimension
        latent_dim: latent space dimension
        hidden_dim: hidden layer dimension
        seq_len: expected sequence length (used for shape validation and sampling)
    Input:
        x: (batch_size, seq_len, feature_dim) time series
    Output:
        x_recon: (batch_size, seq_len, feature_dim) reconstructed time series
        mu: (batch_size, latent_dim, compressed_seq_len) latent distribution mean
        logvar: (batch_size, latent_dim, compressed_seq_len) latent distribution log-variance
    """

    def __init__(self, feature_dim, latent_dim=64, hidden_dim=128, seq_len=None):
        super(TCVAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder: (B, F, T) -> (B, hidden, T/4)
        # achieves 4x temporal compression via two stride=2 convolutions
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # VAE latent-space projection layers
        # 1x1 convolutions project the hidden representation into the latent space
        self.fc_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        # Decoder input layer: (B, latent, T/4) -> (B, hidden, T/4)
        self.decoder_input = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Decoder: (B, hidden, T/4) -> (B, F, T)
        # upsamples via transposed convolutions to restore the original temporal resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            # choose the output activation based on the data range:
            # nn.Sigmoid()  # use when data is in [0, 1]
            # nn.Tanh()     # use when data is in [-1, 1]
        )

    def encode(self, x):
        """
        Encode the input into latent distribution parameters.
        Args:
            x: (B, T, F) input time series
        Returns:
            mu: (B, latent_dim, T') latent distribution mean
            logvar: (B, latent_dim, T') latent distribution log-variance
        """
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) convert to Conv1d format
        h = self.encoder(x)  # (B, hidden_dim, T') encoded hidden representation

        mu = self.fc_mu(h)  # mean
        logvar = self.fc_logvar(h)  # log-variance

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon.
        Lets gradients flow back through the random sampling step.
        Args:
            mu: distribution mean
            logvar: distribution log-variance
        Returns:
            z: the sampled latent variable
        """
        std = torch.exp(0.5 * logvar)  # std = exp(0.5 * log(var))
        eps = torch.randn_like(std)  # sample from a standard normal
        return mu + eps * std  # reparameterized sample

    def decode(self, z):
        """
        Decode a latent variable into a reconstructed sequence.
        Args:
            z: (B, latent_dim, T') latent variable
        Returns:
            x_recon: (B, T, F) reconstructed time series
        """
        h = self.decoder_input(z)  # decoder input processing
        x_recon = self.decoder(h)  # (B, F, T) decoded reconstruction
        return x_recon.transpose(1, 2)  # (B, T, F) back to the original format

    def forward(self, x):
        """
        VAE forward pass.
        Args:
            x: (B, T, F) input time series
        Returns:
            x_recon: (B, T, F) reconstructed time series
            mu: (B, latent_dim, T') latent mean
            logvar: (B, latent_dim, T') latent log-variance
        """
        mu, logvar = self.encode(x)  # encode
        z = self.reparameterize(mu, logvar)  # sample
        x_recon = self.decode(z)  # decode

        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar, kl_weight=1.0):
        """
        VAE loss = reconstruction loss + KL divergence.
        Args:
            x: original input
            x_recon: reconstructed input
            mu: latent mean
            logvar: latent log-variance
            kl_weight: weight on the KL term (the beta of a beta-VAE)
        Returns:
            total_loss: total loss
            recon_loss: reconstruction loss
            kl_loss: KL-divergence loss
        """
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # measures how far the encoded distribution is from a standard normal
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """
        Sample new sequences from the prior distribution.
        Args:
            num_samples: number of samples to generate
            device: device to generate on
        Returns:
            samples: (num_samples, seq_len, feature_dim) generated time series
        """
        # need to know the compressed sequence length
        if self.seq_len is None:
            raise ValueError("seq_len must be specified at init time to sample")

        # compute the compressed length (assuming a compression ratio of 4)
        compressed_len = self.seq_len // 4

        # sample from a standard normal
        z = torch.randn(num_samples, self.latent_dim, compressed_len).to(device)

        # decode to generate samples
        samples = self.decode(z)
        return samples


class TCVAEConceptEncoder(nn.Module):
    """
    A TCVAE-based concept encoder with support for attention pooling.
    Maps a variable-length time series to a fixed-length concept vector.
    Args:
        feature_dim: input feature dimension
        latent_dim: latent space dimension
        hidden_dim: hidden layer dimension
        pooling: pooling strategy ('mean', 'max', 'attention', 'last')
        use_variance: whether to also return a variance estimate
    """

    def __init__(self, feature_dim, latent_dim=64, hidden_dim=128,
                 pooling='attention', use_variance=False):
        super(TCVAEConceptEncoder, self).__init__()

        self.tcv_encoder = TCVAE(feature_dim, latent_dim, hidden_dim)
        self.pooling = pooling
        self.use_variance = use_variance

        # Attention pooling mechanism
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.Tanh(),
                nn.Linear(latent_dim // 2, 1)
            )

    def forward(self, x, return_distribution=False):
        """
        Extract a concept representation from a time series.
        Args:
            x: (B, T, F) time series input
            return_distribution: whether to return mu and logvar separately
        Returns:
            concept: (B, latent_dim) concept representation vector
            or (concept_mu, concept_logvar) if return_distribution=True
        """
        mu, logvar = self.tcv_encoder.encode(x)  # (B, latent_dim, T')

        if return_distribution:
            # return the distribution parameters separately
            if self.pooling == 'mean':
                concept_mu = mu.mean(dim=2)
                concept_logvar = logvar.mean(dim=2)
            elif self.pooling == 'max':
                concept_mu = mu.max(dim=2)[0]
                concept_logvar = logvar.max(dim=2)[0]
            elif self.pooling == 'last':
                concept_mu = mu[:, :, -1]
                concept_logvar = logvar[:, :, -1]
            elif self.pooling == 'attention':
                # attention pooling over the time dimension
                mu_t = mu.transpose(1, 2)  # (B, T', latent_dim)
                attn_weights = F.softmax(self.attention(mu_t), dim=1)  # (B, T', 1)
                concept_mu = (mu_t * attn_weights).sum(dim=1)  # (B, latent_dim)

                logvar_t = logvar.transpose(1, 2)
                concept_logvar = (logvar_t * attn_weights).sum(dim=1)
            else:
                raise ValueError(f"unknown pooling strategy: {self.pooling}")

            return concept_mu, concept_logvar

        else:
            # pool after sampling
            z = self.tcv_encoder.reparameterize(mu, logvar)  # (B, latent_dim, T')

            # temporal pooling
            if self.pooling == 'mean':
                concept = z.mean(dim=2)
            elif self.pooling == 'max':
                concept = z.max(dim=2)[0]
            elif self.pooling == 'last':
                concept = z[:, :, -1]
            elif self.pooling == 'attention':
                z_t = z.transpose(1, 2)  # (B, T', latent_dim)
                attn_weights = F.softmax(self.attention(z_t), dim=1)  # (B, T', 1)
                concept = (z_t * attn_weights).sum(dim=1)  # (B, latent_dim)
            else:
                raise ValueError(f"unknown pooling strategy: {self.pooling}")

            if self.use_variance:
                # also return a variance estimate
                temporal_variability = z.var(dim=2)
                return concept, temporal_variability

            return concept

