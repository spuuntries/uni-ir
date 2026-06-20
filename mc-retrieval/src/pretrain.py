"""Masked Voxel Modeling (MVM) pretraining for the voxel encoder.

Masks ~20% of non-air blocks and trains a U-Net-style encoder-decoder
to reconstruct them. The encoder shares architecture with VoxelEncoder
so weights transfer directly.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import build_block_mapping, remap_voxel
from utils import load_config, set_seed, get_device, save_checkpoint
from model import DepthwiseSeparableConv3d
import spconv.pytorch as spconv


# ---------------------------------------------------------------------------
# Dataset (voxel-only, no text needed)
# ---------------------------------------------------------------------------


class VoxelOnlyDataset(Dataset):
    """Dataset that returns only voxel grids for self-supervised pretraining."""

    def __init__(
        self,
        df: pd.DataFrame,
        block_mapping: dict,
        crop_bbox: bool = True,
        augment: bool = False,
        aug_apply_prob: float = 0.5,
        aug_dropout_prob: float = 0.05,
    ):
        self.voxels = df["voxel_data"].tolist()
        self.block_mapping = block_mapping
        self.crop_bbox = crop_bbox
        self.augment = augment
        self.aug_apply_prob = aug_apply_prob
        self.aug_dropout_prob = aug_dropout_prob

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        voxel = remap_voxel(
            self.voxels[idx], self.block_mapping, crop_bbox=self.crop_bbox
        )
        if self.augment:
            import random

            # 1. Random 90-degree rotations in the horizontal plane (assuming axes 0 and 2 are X and Z)
            k = random.randint(0, 3)
            if k > 0:
                voxel = torch.rot90(voxel, k, [0, 2])

            # 2. Block dropout
            if random.random() < self.aug_apply_prob:
                non_air_mask = voxel != 0
                drop_mask = (
                    torch.rand_like(voxel, dtype=torch.float) < self.aug_dropout_prob
                )
                voxel[non_air_mask & drop_mask] = 0
        return voxel


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def create_mask(voxels: torch.LongTensor, mask_ratio: float = 0.2):
    """Randomly mask non-air blocks.

    Args:
        voxels: (B, 32, 32, 32) block IDs
        mask_ratio: fraction of non-air blocks to mask

    Returns:
        masked_voxels: (B, 32, 32, 32) with masked positions set to mask_token_id
        mask: (B, 32, 32, 32) bool tensor indicating masked positions
    """
    non_air = voxels != 0  # (B, 32, 32, 32)
    rand = torch.rand_like(voxels, dtype=torch.float32)
    mask = non_air & (rand < mask_ratio)
    return mask


# ---------------------------------------------------------------------------
# Model: U-Net style encoder-decoder
# ---------------------------------------------------------------------------


class MaskedVoxelModel(nn.Module):
    """U-Net encoder-decoder for masked block prediction.

    Encoder architecture matches VoxelEncoder exactly so weights
    can be transferred after pretraining.
    """

    def __init__(
        self,
        num_block_types: int = 256,
        block_embed_dim: int = 32,
        channels: list[int] = [64, 128, 256],
        dropout: float = 0.3,
        mask_ratio: float = 0.2,
        use_learned_stem: bool = False,
        use_depthwise_separable: bool = False,
        use_depthwise_separable_decoder: bool = False,
    ):
        super().__init__()
        self.num_block_types = num_block_types
        self.mask_token_id = num_block_types  # extra token for [MASK]
        self.mask_ratio = mask_ratio
        self.use_learned_stem = use_learned_stem

        # +1 for mask token
        self.block_embedding = nn.Embedding(num_block_types + 1, block_embed_dim)

        conv_enc_cls = (
            DepthwiseSeparableConv3d if use_depthwise_separable else nn.Conv3d
        )
        conv_dec_cls = (
            DepthwiseSeparableConv3d if use_depthwise_separable_decoder else nn.Conv3d
        )

        # --- Encoder (mirrors VoxelEncoder.conv_stack) ---
        if use_learned_stem:
            self.stem = nn.Sequential(
                nn.Conv3d(block_embed_dim, block_embed_dim, 4, stride=2, padding=1),
                nn.BatchNorm3d(block_embed_dim),
                nn.GELU(),
            )
        enc_in = block_embed_dim

        # Block 1
        self.enc1 = nn.Sequential(
            conv_enc_cls(enc_in, channels[0], 3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool1 = nn.MaxPool3d(2)

        # Block 2
        self.enc2 = nn.Sequential(
            conv_enc_cls(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck (no pooling)
        self.bottleneck = nn.Sequential(
            conv_enc_cls(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

        # --- Decoder ---
        # Up 2: concat with enc2
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            conv_dec_cls(channels[1] * 2, channels[1], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
        )

        # Up 1: concat with enc1
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            conv_dec_cls(channels[0] * 2, channels[0], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
        )

        if use_learned_stem:
            self.up_stem = nn.ConvTranspose3d(channels[0], block_embed_dim, 2, stride=2)
            self.dec_stem = nn.Sequential(
                conv_dec_cls(block_embed_dim * 2, block_embed_dim, 3, padding=1),
                nn.BatchNorm3d(block_embed_dim),
                nn.GELU(),
            )
            self.pred_head = nn.Conv3d(block_embed_dim, num_block_types, 1)
        else:
            self.pred_head = nn.Conv3d(channels[0], num_block_types, 1)

    def forward(self, voxels: torch.LongTensor):
        """
        Args:
            voxels: (B, 32, 32, 32) original block IDs
        Returns:
            logits: (B, num_blocks, 32, 32, 32) predictions
            mask:   (B, 32, 32, 32) bool mask of what was masked
        """
        # Create mask and apply
        mask = create_mask(voxels, self.mask_ratio)
        masked_voxels = voxels.clone()
        masked_voxels[mask] = self.mask_token_id

        # Embed
        x = self.block_embedding(masked_voxels)  # (B, 32, 32, 32, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, 32, 32, 32)

        if self.use_learned_stem:
            e_in = self.stem(x)
        else:
            e_in = x

        # Encoder
        e1 = self.enc1(e_in)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))

        # Decoder with skip connections
        d2 = self.up2(bn)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        if self.use_learned_stem:
            d_stem = self.up_stem(d1)
            d_stem = self.dec_stem(torch.cat([d_stem, x], dim=1))
            logits = self.pred_head(d_stem)
        else:
            logits = self.pred_head(d1)

        return logits, mask

    def get_encoder_state_dict(self):
        """Extract encoder weights in VoxelEncoder-compatible format."""
        state = {}

        # Block embedding (drop mask token)
        state["block_embedding.weight"] = self.block_embedding.weight[
            : self.num_block_types
        ].clone()

        offset = 0
        if self.use_learned_stem:
            stem_conv = self.stem[0]
            stem_bn = self.stem[1]
            state["conv_stack.0.weight"] = stem_conv.weight.clone()
            state["conv_stack.0.bias"] = stem_conv.bias.clone()
            state["conv_stack.1.weight"] = stem_bn.weight.clone()
            state["conv_stack.1.bias"] = stem_bn.bias.clone()
            state["conv_stack.1.running_mean"] = stem_bn.running_mean.clone()
            state["conv_stack.1.running_var"] = stem_bn.running_var.clone()
            state["conv_stack.1.num_batches_tracked"] = (
                stem_bn.num_batches_tracked.clone()
            )
            offset = 3

        mapping = {
            "enc1": offset + 0,
            "enc2": offset + 5,
            "bottleneck": offset + 10,
        }

        for block_name, stack_offset in mapping.items():
            block = getattr(self, block_name)
            conv = block[0]

            if isinstance(conv, DepthwiseSeparableConv3d):
                state[f"conv_stack.{stack_offset}.depthwise.weight"] = (
                    conv.depthwise.weight.clone()
                )
                state[f"conv_stack.{stack_offset}.depthwise.bias"] = (
                    conv.depthwise.bias.clone()
                )
                state[f"conv_stack.{stack_offset}.pointwise.weight"] = (
                    conv.pointwise.weight.clone()
                )
                state[f"conv_stack.{stack_offset}.pointwise.bias"] = (
                    conv.pointwise.bias.clone()
                )
            else:
                state[f"conv_stack.{stack_offset}.weight"] = conv.weight.clone()
                state[f"conv_stack.{stack_offset}.bias"] = conv.bias.clone()

            bn = block[1]
            state[f"conv_stack.{stack_offset + 1}.weight"] = bn.weight.clone()
            state[f"conv_stack.{stack_offset + 1}.bias"] = bn.bias.clone()
            state[f"conv_stack.{stack_offset + 1}.running_mean"] = (
                bn.running_mean.clone()
            )
            state[f"conv_stack.{stack_offset + 1}.running_var"] = bn.running_var.clone()
            state[f"conv_stack.{stack_offset + 1}.num_batches_tracked"] = (
                bn.num_batches_tracked.clone()
            )

        return state


# ---------------------------------------------------------------------------
# Sparse Masked Voxel Model (spconv)
# ---------------------------------------------------------------------------


class SparseMaskedVoxelModel(nn.Module):
    def __init__(
        self,
        num_block_types: int = 256,
        block_embed_dim: int = 32,
        channels: list[int] = [64, 128, 256],
        mask_ratio: float = 0.2,
    ):
        super().__init__()
        self.num_block_types = num_block_types
        self.mask_token_id = num_block_types
        self.mask_ratio = mask_ratio
        self.channels = channels

        self.block_embedding = nn.Embedding(num_block_types + 1, block_embed_dim)

        self.encoder_blocks = nn.ModuleList()
        in_ch = block_embed_dim
        for i, out_ch in enumerate(channels):
            self.encoder_blocks.append(
                nn.ModuleDict(
                    {
                        "subm": spconv.SubMConv3d(
                            in_ch,
                            out_ch,
                            3,
                            padding=1,
                            bias=False,
                            indice_key=f"subm{i}",
                        ),
                        "subm_bn": nn.BatchNorm1d(out_ch),
                        "down": spconv.SparseConv3d(
                            out_ch,
                            out_ch,
                            3,
                            stride=2,
                            padding=1,
                            bias=False,
                            indice_key=f"down{i}",
                        ),
                        "down_bn": nn.BatchNorm1d(out_ch),
                    }
                )
            )
            in_ch = out_ch

        self.bottleneck = spconv.SubMConv3d(
            in_ch, in_ch, 3, padding=1, bias=False, indice_key=f"subm{len(channels)}"
        )
        self.bottleneck_bn = nn.BatchNorm1d(in_ch)

        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(len(channels))):
            skip_ch = channels[i]
            out_ch = channels[i - 1] if i > 0 else block_embed_dim
            self.decoder_blocks.append(
                nn.ModuleDict(
                    {
                        "up": spconv.SparseInverseConv3d(
                            in_ch, skip_ch, 3, bias=False, indice_key=f"down{i}"
                        ),
                        "dec": spconv.SubMConv3d(
                            skip_ch * 2,
                            out_ch,
                            3,
                            padding=1,
                            bias=False,
                            indice_key=f"subm{i}",
                        ),
                        "dec_bn": nn.BatchNorm1d(out_ch),
                    }
                )
            )
            in_ch = out_ch

        self.pred_head = nn.Linear(block_embed_dim, num_block_types)
        self.gelu = nn.GELU()

    def forward(self, voxels: torch.LongTensor):
        batch_size = voxels.shape[0]

        # 1. Find non-air coordinates
        non_air_coords = torch.nonzero(voxels != 0)

        # 2. Extract their block IDs
        block_ids = voxels[
            non_air_coords[:, 0],
            non_air_coords[:, 1],
            non_air_coords[:, 2],
            non_air_coords[:, 3],
        ]

        # 3. Create boolean mask for THESE non-air blocks
        num_non_air = len(block_ids)
        rand = torch.rand(num_non_air, device=voxels.device)
        is_masked = rand < self.mask_ratio

        # 4. Replace masked IDs with mask_token_id
        masked_block_ids = block_ids.clone()
        masked_block_ids[is_masked] = self.mask_token_id

        # 5. Embed
        features = self.block_embedding(masked_block_ids)

        # 6. Create Sparse Tensor
        coords = non_air_coords.to(torch.int32)
        spatial_shape = voxels.shape[1:]
        x = spconv.SparseConvTensor(features, coords, spatial_shape, batch_size)

        # --- Encoder ---
        encoder_features = [x]
        curr = x
        for block in self.encoder_blocks:
            subm = block["subm"](curr)
            subm = subm.replace_feature(self.gelu(block["subm_bn"](subm.features)))
            curr = block["down"](subm)
            curr = curr.replace_feature(self.gelu(block["down_bn"](curr.features)))
            encoder_features.append(subm)

        # --- Bottleneck ---
        curr = self.bottleneck(curr)
        curr = curr.replace_feature(self.gelu(self.bottleneck_bn(curr.features)))

        # --- Decoder ---
        for i, block in enumerate(self.decoder_blocks):
            up = block["up"](curr)
            skip = encoder_features[-(i + 1)]
            cat_features = torch.cat([up.features, skip.features], dim=1)
            cat = up.replace_feature(cat_features)
            curr = block["dec"](cat)
            curr = curr.replace_feature(self.gelu(block["dec_bn"](curr.features)))

        # Create a dense boolean mask of masked blocks
        dense_mask = torch.zeros_like(voxels, dtype=torch.bool)
        dense_mask[
            non_air_coords[:, 0],
            non_air_coords[:, 1],
            non_air_coords[:, 2],
            non_air_coords[:, 3],
        ] = is_masked

        # Indices of the sparse tensor in their current scrambled order
        inds = curr.indices.long()

        # Query the dense mask to find which of the scrambled features correspond to masked blocks
        scrambled_is_masked = dense_mask[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

        # Compute logits on the active sparse features (N, 512)
        logits = self.pred_head(curr.features)

        # Extract the logits for the masked blocks (M, 512)
        masked_logits = logits[scrambled_is_masked]

        # Extract the true labels for the masked blocks from the original voxels tensor (M,)
        masked_coords = inds[scrambled_is_masked]
        masked_labels = voxels[
            masked_coords[:, 0],
            masked_coords[:, 1],
            masked_coords[:, 2],
            masked_coords[:, 3],
        ]

        return masked_logits, masked_labels

    def get_encoder_state_dict(self):
        state = {}
        state["block_embedding.weight"] = self.block_embedding.weight[
            : self.num_block_types
        ].clone()

        offset = 0
        for i, block in enumerate(self.encoder_blocks):
            for name, seq_idx in [
                ("subm", 0),
                ("subm_bn", 1),
                ("down", 3),
                ("down_bn", 4),
            ]:
                mod = block[name]
                for k, v in mod.state_dict().items():
                    state[f"sparse_conv_stack.{offset + seq_idx}.{k}"] = v.clone()
            offset += 6

        return state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def pretrain(cfg: dict):
    """Run masked voxel modeling pretraining."""
    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    pt_cfg = cfg.get("pretraining", {})
    mask_ratio = pt_cfg.get("mask_ratio", 0.2)
    epochs = pt_cfg.get("epochs", 200)
    batch_size = pt_cfg.get("batch_size", 256)
    lr = pt_cfg.get("lr", 1e-3)
    ckpt_dir = pt_cfg.get("checkpoint_dir", "checkpoints")

    # --- data (use ALL samples, no splits needed) ---
    df = pd.read_parquet(cfg["data"]["parquet_path"])
    print(f"Loaded {len(df)} samples for pretraining")

    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )
    num_blocks = cfg["data"]["max_block_types"]

    crop_bbox = cfg["data"].get("crop_bbox", True)
    augment = pt_cfg.get("augment", cfg["data"].get("augment", True))
    aug_apply_prob = cfg["data"].get("aug_apply_prob", 0.5)
    aug_dropout_prob = cfg["data"].get("aug_dropout_prob", 0.05)

    dataset = VoxelOnlyDataset(
        df,
        block_mapping,
        crop_bbox=crop_bbox,
        augment=augment,
        aug_apply_prob=aug_apply_prob,
        aug_dropout_prob=aug_dropout_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --- model ---
    model_cfg = cfg["model"]
    use_sparse = model_cfg.get("use_sparse", False)

    if use_sparse:
        model = SparseMaskedVoxelModel(
            num_block_types=num_blocks,
            block_embed_dim=model_cfg["block_embed_dim"],
            channels=model_cfg["voxel_channels"],
            mask_ratio=mask_ratio,
        ).to(device)
    else:
        model = MaskedVoxelModel(
            num_block_types=num_blocks,
            block_embed_dim=model_cfg["block_embed_dim"],
            channels=model_cfg["voxel_channels"],
            dropout=model_cfg.get("dropout", 0.3),
            mask_ratio=mask_ratio,
            use_learned_stem=model_cfg.get("use_learned_stem", False),
            use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
            use_depthwise_separable_decoder=model_cfg.get(
                "use_depthwise_separable_decoder", False
            ),
        ).to(device)

    if model_cfg.get("semantic_init", False):
        from dataset import extract_block_names
        from model import TextEncoder, apply_semantic_init

        block_names = extract_block_names(df, block_mapping)
        # mask token gets an arbitrary string "mask token"
        block_names.append("mask token")

        temp_text_encoder = TextEncoder(
            model_name=model_cfg["text_model"],
            text_hidden_dim=model_cfg["text_hidden_dim"],
            embed_dim=model_cfg["embed_dim"],
            freeze=True,
        ).to(device)

        apply_semantic_init(
            voxel_embedding_layer=model.block_embedding,
            text_encoder=temp_text_encoder,
            block_names=block_names,
            block_embed_dim=model_cfg["block_embed_dim"],
            device=device,
        )
        del temp_text_encoder
        torch.cuda.empty_cache()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"MVM model parameters: {param_count:,}")

    # --- optimizer ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- training ---
    print(f"\nStarting MVM pretraining for {epochs} epochs...")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Batch size: {batch_size}")
    print()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"  epoch {epoch:3d}", leave=False)
        for batch_idx, voxels in enumerate(pbar):
            voxels = voxels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, labels = model(voxels)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # compute training accuracy (just for logging)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                n_correct = (preds == labels).sum().item()
                n_masked = len(labels)

            total_loss += loss.item()
            total_correct += n_correct
            total_masked += n_masked
            num_batches += 1

            acc = n_correct / max(n_masked, 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_correct / max(total_masked, 1)
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  mask_acc={avg_acc:.4f}  "
            f"lr={lr_current:.2e}"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder_state": model.get_encoder_state_dict(),
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "cfg": cfg,
                },
                os.path.join(ckpt_dir, "pretrained_voxel.pt"),
            )

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain voxel encoder via masked voxel modeling"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pretrain(cfg)
