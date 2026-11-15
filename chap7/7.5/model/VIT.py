import torch
import torch.nn as nn
import torch.nn.functional as F


#================(3) ViT for Classification ========================
class ViTClassification(nn.Module):
    """
    Internally adaptively pools the image to 8x8 first, then does patch
    embedding with patch_size=2 to get 16 patch tokens, adds a cls token,
    and feeds the result into a Transformer Encoder for classification.
    """
    def __init__(self,num_classes, img_pool_size, patch_size,  embed_dim, num_layers, nheads, mlp_ratio, dropout):
        super().__init__()
        self.img_pool_size = img_pool_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # ---- Patch Embedding: pool to a fixed size first, then split into patches with a Conv ----
        self.pool = nn.AdaptiveAvgPool2d(img_pool_size)         # [B,1,9,11] -> [B,1,8,8]
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)  # [B,embed,4,4]
        num_patches = (img_pool_size[0] // patch_size) * (img_pool_size[1] // patch_size)  # 16

        # ---- Learnable CLS token + 1D positional encoding (supports length interpolation) ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))  # [CLS + patches, dim]
        self.pos_drop = nn.Dropout(dropout)

        # ---- Transformer Encoder (batch_first=True, to avoid a warning) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nheads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- Classification head ----
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _interpolate_pos_embed(self, x):
        """
        When the number of tokens changes, linearly interpolate the
        non-CLS positional encodings to match the new length.
        x: [B, N, C]  (N = 1 + num_patches)
        """
        B, N, C = x.shape
        if self.pos_embed.shape[1] == N:
            return self.pos_embed

        # Split into the CLS and patch parts
        pos_cls = self.pos_embed[:, :1, :]              # [1,1,C]
        pos_patch = self.pos_embed[:, 1:, :]            # [1,L,C]
        L_old = pos_patch.shape[1]
        L_new = N - 1

        # 1D linear interpolation to L_new
        pos_patch = pos_patch.transpose(1, 2)           # [1,C,L_old]
        pos_patch = F.interpolate(pos_patch, size=L_new, mode="linear", align_corners=False)
        pos_patch = pos_patch.transpose(1, 2)           # [1,L_new,C]
        return torch.cat([pos_cls, pos_patch], dim=1)   # [1,N,C]

    def forward(self, x):
        # x: [B,1,9,11]
        B = x.size(0)
        x = self.pool(x)                                # [B,1,8,8]
        x = self.proj(x)                                # [B,embed,4,4]
        x = x.flatten(2).transpose(1, 2)                # [B,16,embed]

        # Add the CLS token
        cls_tok = self.cls_token.expand(B, -1, -1)      # [B,1,embed]
        x = torch.cat([cls_tok, x], dim=1)              # [B,17,embed]

        # Add the positional encoding (interpolated if needed)
        pos = self._interpolate_pos_embed(x)
        x = self.pos_drop(x + pos)

        # Transformer Encoder
        x = self.transformer(x)                         # [B,17,embed]

        # Take the CLS vector -> classify
        x = self.norm(x[:, 0])                          # [B,embed]
        return self.head(x)                             # [B,num_classes]
