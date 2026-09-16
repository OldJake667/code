"""Small, readable Vision Transformer construction used by ViT-Tiny and DINOv2.

The architecture is built here with ordinary PyTorch layers.  timm is not used
to construct either model.  Pretrained checkpoint tensors are loaded separately.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Cut an image into non-overlapping patches and embed every patch."""

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        patches = self.proj(images)          # B, features, rows, columns
        patches = patches.flatten(2)         # B, features, number of patches
        return patches.transpose(1, 2)       # B, number of patches, features


class Attention(nn.Module):
    """Multi-head self-attention written explicitly instead of nn.MultiheadAttention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        batch, token_count, embed_dim = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(batch, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        mixed = attention @ value
        mixed = mixed.transpose(1, 2).reshape(batch, token_count, embed_dim)
        return self.proj(mixed)


class Mlp(nn.Module):
    """The two linear layers inside each Transformer block."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, tokens):
        return self.fc2(self.act(self.fc1(tokens)))


class LayerScale(nn.Module):
    """DINOv2 learns a small scale for each residual branch."""

    def __init__(self, embed_dim, initial_value):
        super().__init__()
        self.gamma = nn.Parameter(initial_value * torch.ones(embed_dim))

    def forward(self, tokens):
        return tokens * self.gamma


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, layer_scale=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(embed_dim, num_heads)
        self.ls1 = LayerScale(embed_dim, layer_scale) if layer_scale else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))
        self.ls2 = LayerScale(embed_dim, layer_scale) if layer_scale else nn.Identity()

    def forward(self, tokens):
        tokens = tokens + self.ls1(self.attn(self.norm1(tokens)))
        tokens = tokens + self.ls2(self.mlp(self.norm2(tokens)))
        return tokens


class VisionTransformer(nn.Module):
    """The complete classifier: patches -> Transformer blocks -> two-class head."""

    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        num_classes=2,
        layer_scale=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        token_count = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, layer_scale=layer_scale)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, images):
        patches = self.patch_embed(images)
        class_token = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_token, patches), dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(self, images):
        return self.head(self.forward_features(images))


def build_vit_tiny():
    """ViT-Tiny/16: 192 features, 12 blocks and 3 attention heads."""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )


def build_dinov2_small():
    """The ViT-S/14 architecture used by the DINOv2 small checkpoint."""
    return VisionTransformer(
        image_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        layer_scale=1.0,
    )


def _load_except_head(model, state):
    state = dict(state)
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    result = model.load_state_dict(state, strict=False)
    expected_missing = {"head.weight", "head.bias"}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def load_vit_tiny_pretrained_backbone(model, checkpoint_path):
    """Load the unchanged pretrained backbone from the earlier ViT checkpoint."""
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    _load_except_head(model, state)


def _resize_dinov2_position_embedding(position_embedding, target_grid=16):
    """Resize DINOv2's pretraining grid to the 224/14 = 16 by 16 grid."""
    class_position = position_embedding[:, :1]
    patch_positions = position_embedding[:, 1:]
    old_grid = int(patch_positions.shape[1] ** 0.5)
    patch_positions = patch_positions.reshape(1, old_grid, old_grid, -1)
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = F.interpolate(
        patch_positions,
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, target_grid**2, -1)
    return torch.cat((class_position, patch_positions), dim=1)


def load_dinov2_pretrained_backbone(model, checkpoint_path):
    """Load Meta's DINOv2 ViT-S/14 checkpoint into our explicit architecture."""
    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = dict(state.get("model", state))
    state.pop("mask_token", None)  # Used during DINOv2 pretraining, not classification.
    state["pos_embed"] = _resize_dinov2_position_embedding(state["pos_embed"])
    _load_except_head(model, state)
