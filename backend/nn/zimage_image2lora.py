"""
Z-Image Image-to-LoRA Model

Converts image embeddings from SigLIP2 + DINOv3 encoders into LoRA weights
that can be applied to the Z-Image transformer for style transfer.

Input: 5632-dim embedding (1536 from SigLIP2 + 4096 from DINOv3)
Output: Dict of LoRA weights for Z-Image transformer layers
"""

import torch
import torch.nn as nn


class CompressedMLP(nn.Module):
    """Simple MLP with optional residual input."""

    def __init__(self, in_dim, mid_dim, out_dim, bias=False):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, mid_dim, bias=bias)
        self.proj_out = nn.Linear(mid_dim, out_dim, bias=bias)

    def forward(self, x, residual=None):
        x = self.proj_in(x)
        if residual is not None:
            x = x + residual
        x = self.proj_out(x)
        return x


class SequentialMLP(nn.Module):
    """Sequential MLP that flattens input sequence before projection."""

    def __init__(self, length, in_dim, mid_dim, out_dim, bias=False):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, mid_dim, bias=bias)
        self.proj_out = nn.Linear(length * mid_dim, out_dim, bias=bias)
        self.length = length
        self.in_dim = in_dim
        self.mid_dim = mid_dim

    def forward(self, x):
        x = x.view(self.length, self.in_dim)
        x = self.proj_in(x)
        x = x.view(1, self.length * self.mid_dim)
        x = self.proj_out(x)
        return x


class ImageEmbeddingToLoraMatrix(nn.Module):
    """Converts image embedding to LoRA A and B matrices."""

    def __init__(self, in_dim, compress_dim, lora_a_dim, lora_b_dim, rank):
        super().__init__()
        self.proj_a = CompressedMLP(in_dim, compress_dim, lora_a_dim * rank)
        self.proj_b = CompressedMLP(in_dim, compress_dim, lora_b_dim * rank)
        self.lora_a_dim = lora_a_dim
        self.lora_b_dim = lora_b_dim
        self.rank = rank

    def forward(self, x, residual=None):
        lora_a = self.proj_a(x, residual).view(self.rank, self.lora_a_dim)
        lora_b = self.proj_b(x, residual).view(self.lora_b_dim, self.rank)
        return lora_a, lora_b


class LoRATrainerBlock(nn.Module):
    """Generates LoRA weights for a single transformer block."""

    def __init__(
        self,
        lora_patterns,
        in_dim=1536 + 4096,
        compress_dim=128,
        rank=4,
        block_id=0,
        use_residual=True,
        residual_length=64 + 7,
        residual_dim=3584,
        residual_mid_dim=1024,
        prefix="transformer_blocks"
    ):
        super().__init__()
        self.prefix = prefix
        self.lora_patterns = lora_patterns
        self.block_id = block_id
        self.layers = nn.ModuleList([
            ImageEmbeddingToLoraMatrix(in_dim, compress_dim, lora_a_dim, lora_b_dim, rank)
            for name, lora_a_dim, lora_b_dim in lora_patterns
        ])
        if use_residual:
            self.proj_residual = SequentialMLP(residual_length, residual_dim, residual_mid_dim, compress_dim)
        else:
            self.proj_residual = None

    def forward(self, x, residual=None):
        lora = {}
        if self.proj_residual is not None:
            residual = self.proj_residual(residual)
        for lora_pattern, layer in zip(self.lora_patterns, self.layers):
            name = lora_pattern[0]
            lora_a, lora_b = layer(x, residual=residual)
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_A.default.weight"] = lora_a
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_B.default.weight"] = lora_b
        return lora


class ZImageImage2LoRAComponent(nn.Module):
    """Generates LoRA weights for a component (layers, context_refiner, noise_refiner)."""

    def __init__(
        self,
        lora_patterns,
        prefix,
        num_blocks=60,
        use_residual=True,
        compress_dim=128,
        rank=4,
        residual_length=64 + 7,
        residual_mid_dim=1024
    ):
        super().__init__()
        self.lora_patterns = lora_patterns
        self.num_blocks = num_blocks
        blocks = []
        for patterns in lora_patterns:
            for block_id in range(num_blocks):
                blocks.append(LoRATrainerBlock(
                    patterns,
                    block_id=block_id,
                    use_residual=use_residual,
                    compress_dim=compress_dim,
                    rank=rank,
                    residual_length=residual_length,
                    residual_mid_dim=residual_mid_dim,
                    prefix=prefix
                ))
        self.blocks = nn.ModuleList(blocks)
        self.residual_scale = 0.05
        self.use_residual = use_residual

    def forward(self, x, residual=None):
        if residual is not None:
            if self.use_residual:
                residual = residual * self.residual_scale
            else:
                residual = None
        lora = {}
        for block in self.blocks:
            lora.update(block(x, residual))
        return lora


class ZImageImage2LoRAModel(nn.Module):
    """
    Main Image-to-LoRA model for Z-Image.

    Converts concatenated SigLIP2 + DINOv3 embeddings (5632-dim) into LoRA weights
    for the Z-Image transformer. Generates LoRA for:
    - 30 main transformer layers
    - 2 context_refiner layers
    - 2 noise_refiner layers

    Each layer gets LoRA for attention (to_q, to_k, to_v, to_out.0) and
    feed-forward (w1, w2, w3) modules.
    """

    def __init__(
        self,
        use_residual=False,
        compress_dim=64,
        rank=4,
        residual_length=64 + 7,
        residual_mid_dim=1024
    ):
        super().__init__()
        # LoRA patterns: (layer_name, lora_a_dim, lora_b_dim)
        lora_patterns = [
            [
                ("attention.to_q", 3840, 3840),
                ("attention.to_k", 3840, 3840),
                ("attention.to_v", 3840, 3840),
                ("attention.to_out.0", 3840, 3840),
            ],
            [
                ("feed_forward.w1", 3840, 10240),
                ("feed_forward.w2", 10240, 3840),
                ("feed_forward.w3", 3840, 10240),
            ],
        ]
        config = {
            "lora_patterns": lora_patterns,
            "use_residual": use_residual,
            "compress_dim": compress_dim,
            "rank": rank,
            "residual_length": residual_length,
            "residual_mid_dim": residual_mid_dim,
        }
        self.layers_lora = ZImageImage2LoRAComponent(
            prefix="layers",
            num_blocks=30,
            **config,
        )
        self.context_refiner_lora = ZImageImage2LoRAComponent(
            prefix="context_refiner",
            num_blocks=2,
            **config,
        )
        self.noise_refiner_lora = ZImageImage2LoRAComponent(
            prefix="noise_refiner",
            num_blocks=2,
            **config,
        )

    def forward(self, x, residual=None):
        """
        Generate LoRA weights from image embeddings.

        Args:
            x: Image embedding tensor of shape [batch, 5632]
            residual: Optional residual features

        Returns:
            Dict of LoRA weights with keys like:
            - "layers.0.attention.to_q.lora_A.default.weight"
            - "layers.0.attention.to_q.lora_B.default.weight"
            etc.
        """
        lora = {}
        lora.update(self.layers_lora(x, residual=residual))
        lora.update(self.context_refiner_lora(x, residual=residual))
        lora.update(self.noise_refiner_lora(x, residual=residual))
        return lora

    def initialize_weights(self):
        """Initialize weights for training (not needed for inference)."""
        state_dict = self.state_dict()
        for name in state_dict:
            if ".proj_a." in name:
                state_dict[name] = state_dict[name] * 0.3
            elif ".proj_b.proj_out." in name:
                state_dict[name] = state_dict[name] * 0
            elif ".proj_residual.proj_out." in name:
                state_dict[name] = state_dict[name] * 0.3
        self.load_state_dict(state_dict)


def load_zimage_i2l_model(model_path: str, device="cpu", dtype=torch.bfloat16):
    """
    Load the Z-Image Image2LoRA model from a safetensors file.

    Args:
        model_path: Path to model.safetensors
        device: Device to load to
        dtype: Data type for weights

    Returns:
        Loaded ZImageImage2LoRAModel
    """
    from safetensors.torch import load_file

    # The i2L model uses compress_dim=128 (not 64)
    model = ZImageImage2LoRAModel(
        use_residual=False,
        compress_dim=128,
        rank=4,
    )

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def merge_lora_dicts(lora_dicts: list, alphas: list = None) -> dict:
    """
    Merge multiple LoRA dictionaries with optional alpha weighting.

    Args:
        lora_dicts: List of LoRA weight dictionaries
        alphas: Optional list of alpha weights (default: equal weighting)

    Returns:
        Merged LoRA dictionary
    """
    if not lora_dicts:
        return {}

    if alphas is None:
        alphas = [1.0 / len(lora_dicts)] * len(lora_dicts)

    merged = {}
    for key in lora_dicts[0].keys():
        merged[key] = sum(
            lora_dict[key] * alpha
            for lora_dict, alpha in zip(lora_dicts, alphas)
        )

    return merged
