"""
Z-Image i2L Image Encoders

SigLIP2 and DINOv3 image encoders for the Image-to-LoRA pipeline.
These encode style reference images into embeddings that are then
converted to LoRA weights.

Expected model locations:
- models/Z-Image-i2L/SigLIP2-G384/model.safetensors
- models/Z-Image-i2L/DINOv3-7B/model.safetensors
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Union
import os

# Import from transformers
try:
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipVisionConfig
    from transformers import SiglipImageProcessor
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False

try:
    from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTConfig
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False


class Siglip2ImageEncoder(SiglipVisionTransformer):
    """SigLIP2 image encoder for i2L."""

    def __init__(self):
        if not SIGLIP_AVAILABLE:
            raise ImportError("transformers with SigLIP support required")

        config = SiglipVisionConfig(
            attention_dropout=0.0,
            hidden_act="gelu_pytorch_tanh",
            hidden_size=1536,
            image_size=384,
            intermediate_size=6144,
            layer_norm_eps=1e-06,
            model_type="siglip_vision_model",
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=40,
            patch_size=16,
            _attn_implementation="sdpa"
        )
        super().__init__(config)
        self.processor = SiglipImageProcessor(
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            resample=2,
            rescale_factor=0.00392156862745098,
            size={"height": 384, "width": 384}
        )

    def load_weights(self, model_path: str):
        """Load weights from safetensors file."""
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        self.load_state_dict(state_dict)

    def encode(self, image: Image.Image, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """
        Encode an image to SigLIP2 embedding.

        Args:
            image: PIL Image
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor of shape [1, 1536]
        """
        pixel_values = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=False)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Use head for pooling if available
        if self.use_head:
            pooled = self.head(last_hidden_state)
        else:
            pooled = last_hidden_state[:, 0, :]

        return pooled


class DINOv3ImageEncoder(DINOv3ViTModel):
    """DINOv3 image encoder for i2L."""

    def __init__(self):
        if not DINOV3_AVAILABLE:
            raise ImportError("transformers with DINOv3 support required")

        config = DINOv3ViTConfig(
            attention_dropout=0.0,
            drop_path_rate=0.0,
            hidden_act="silu",
            hidden_size=4096,
            image_size=224,
            initializer_range=0.02,
            intermediate_size=8192,
            key_bias=False,
            layer_norm_eps=1e-05,
            layerscale_value=1.0,
            mlp_bias=True,
            model_type="dinov3_vit",
            num_attention_heads=32,
            num_channels=3,
            num_hidden_layers=40,
            num_register_tokens=4,
            patch_size=16,
            pos_embed_rescale=2.0,
            proj_bias=True,
            query_bias=False,
            rope_theta=100.0,
            use_gated_mlp=True,
            value_bias=False
        )
        super().__init__(config)
        self.processor = DINOv3ViTImageProcessorFast(
            data_format="channels_first",
            default_to_square=True,
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            resample=2,
            rescale_factor=0.00392156862745098,
            size={"height": 224, "width": 224}
        )

    def load_weights(self, model_path: str):
        """Load weights from safetensors file."""
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        self.load_state_dict(state_dict)

    def encode(self, image: Image.Image, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """
        Encode an image to DINOv3 embedding.

        Args:
            image: PIL Image
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor of shape [1, 4096]
        """
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

        hidden_states = self.embeddings(pixel_values, bool_masked_pos=None)
        position_embeddings = self.rope_embeddings(pixel_values)

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]  # CLS token

        return pooled_output


class I2LImageEncoder:
    """
    Combined SigLIP2 + DINOv3 encoder for Image-to-LoRA.

    Encodes images and concatenates embeddings to produce 5632-dim vectors
    suitable for the ZImageImage2LoRAModel.
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize the encoder.

        Args:
            models_dir: Path to models directory containing Z-Image-i2L folder.
                       If None, uses models/ relative to chromaforge root.
        """
        self.models_dir = models_dir
        self.siglip2 = None
        self.dinov3 = None
        self._loaded = False

    def _get_model_paths(self):
        """Get paths to encoder model files."""
        if self.models_dir:
            base = self.models_dir
        else:
            # Default to models/Z-Image-i2L/
            base = os.path.join(os.path.dirname(__file__), "..", "..", "models", "Z-Image-i2L")

        siglip_path = os.path.join(base, "SigLIP2-G384", "model.safetensors")
        dinov3_path = os.path.join(base, "DINOv3-7B", "model.safetensors")

        return siglip_path, dinov3_path

    def load(self, device="cuda", dtype=torch.bfloat16):
        """Load encoder models."""
        if self._loaded:
            return

        siglip_path, dinov3_path = self._get_model_paths()

        print(f"Loading SigLIP2 from {siglip_path}...")
        self.siglip2 = Siglip2ImageEncoder()
        self.siglip2.load_weights(siglip_path)
        self.siglip2 = self.siglip2.to(device=device, dtype=dtype)
        self.siglip2.eval()

        print(f"Loading DINOv3 from {dinov3_path}...")
        self.dinov3 = DINOv3ImageEncoder()
        self.dinov3.load_weights(dinov3_path)
        self.dinov3 = self.dinov3.to(device=device, dtype=dtype)
        self.dinov3.eval()

        self._loaded = True

    def unload(self):
        """Unload models to free VRAM."""
        if self.siglip2 is not None:
            del self.siglip2
            self.siglip2 = None
        if self.dinov3 is not None:
            del self.dinov3
            self.dinov3 = None
        self._loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
        device="cuda",
        dtype=torch.bfloat16
    ) -> torch.Tensor:
        """
        Encode images to concatenated SigLIP2 + DINOv3 embeddings.

        Args:
            images: Single PIL Image or list of PIL Images
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor of shape [N, 5632] where N is number of images
        """
        if not self._loaded:
            self.load(device, dtype)

        if isinstance(images, Image.Image):
            images = [images]

        embeddings = []
        for img in images:
            # Encode with both models using encode() method
            siglip_emb = self.siglip2.encode(img, device=device, dtype=dtype)  # [1, 1536]
            dinov3_emb = self.dinov3.encode(img, device=device, dtype=dtype)   # [1, 4096]

            # Concatenate
            combined = torch.cat([siglip_emb, dinov3_emb], dim=-1)  # [1, 5632]
            embeddings.append(combined)

        return torch.cat(embeddings, dim=0)  # [N, 5632]

    def encode_and_unload(
        self,
        images: Union[Image.Image, List[Image.Image]],
        device="cuda",
        dtype=torch.bfloat16
    ) -> torch.Tensor:
        """Encode images and immediately unload models to free VRAM."""
        result = self.encode(images, device, dtype)
        self.unload()
        return result
