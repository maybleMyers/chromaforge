# ChromaRadiance AutoEncoder implementation based on flow repository
# This is the specialized VAE used by ChromaRadiance models for 64-channel latent space processing

import math
import torch
import torch.nn as nn
from torch import Tensor


class ChromaRadianceAutoEncoderConfig:
    """Config object for ChromaRadianceAutoEncoder to be compatible with VAE class."""
    def __init__(self, bottleneck_channels=64, spatial_compression=8):
        self.latent_channels = bottleneck_channels
        self.down_block_types = ['DownBlock'] * int(math.log2(spatial_compression))  # For downscale_ratio calculation


class ChromaRadianceAutoEncoder(nn.Module):
    """
    Specialized autoencoder for ChromaRadiance models.
    Encodes RGB images to 64-channel latent space and decodes back to RGB.
    Based on the flow repository's autoencoder implementation.
    """
    
    def __init__(
        self, 
        pixel_channels: int = 3,
        bottleneck_channels: int = 64,
        spatial_compression: int = 8,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.pixel_channels = pixel_channels
        self.bottleneck_channels = bottleneck_channels  
        self.spatial_compression = spatial_compression
        
        # Create config for VAE compatibility
        self.config = ChromaRadianceAutoEncoderConfig(bottleneck_channels, spatial_compression)
        
        # Encoder: RGB -> 64-channel latent
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(pixel_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            
            # Downsampling blocks (8x spatial compression total)
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),  # /4  
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, bottleneck_channels, 4, stride=2, padding=1),  # /8
            nn.SiLU(),
        )
        
        # Decoder: 64-channel latent -> RGB
        self.decoder = nn.Sequential(
            # Upsampling blocks
            nn.ConvTranspose2d(bottleneck_channels, base_channels * 4, 4, stride=2, padding=1),  # x2
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),  # x4
            nn.SiLU(), 
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),  # x8
            nn.SiLU(),
            
            # Final conv to RGB
            nn.Conv2d(base_channels, pixel_channels, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode RGB image to 64-channel latent space.
        Args:
            x: RGB image tensor [B, 3, H, W] in [0, 1] or [-1, 1] range
        Returns:
            latent: Latent tensor [B, 64, H//8, W//8]
        """
        # Normalize to [-1, 1] if needed
        if x.max() > 1.1:  # Likely in [0, 255] range
            x = x / 255.0
        if x.min() >= 0 and x.max() <= 1.1:  # In [0, 1] range
            x = x * 2.0 - 1.0  # Convert to [-1, 1]
            
        latent = self.encoder(x)
        return latent
        
    def decode(self, latent: Tensor) -> Tensor:
        """
        Decode 64-channel latent to RGB image.
        Args:
            latent: Latent tensor [B, 64, H//8, W//8] 
        Returns:
            image: RGB image tensor [B, 3, H, W] in [0, 1] range
        """
        image = self.decoder(latent)
        # Convert from [-1, 1] to [0, 1]
        image = (image + 1.0) * 0.5
        return image.clamp(0, 1)


def vae_flatten(latent: Tensor) -> tuple[Tensor, tuple]:
    """
    Flatten latent tensor for sequence processing by the transformer.
    Based on flow repository implementation.
    
    Args:
        latent: Latent tensor [B, C, H, W]
    Returns:
        flattened: Flattened tensor [B, H*W, C] for sequence processing
        shape: Original spatial shape (H, W) for unflattening
    """
    B, C, H, W = latent.shape
    flattened = latent.flatten(2).transpose(1, 2)  # [B, H*W, C]
    return flattened, (H, W)


def vae_unflatten(flattened: Tensor, shape: tuple) -> Tensor:
    """
    Unflatten sequence tensor back to spatial format.
    
    Args:
        flattened: Flattened tensor [B, H*W, C]
        shape: Original spatial shape (H, W)
    Returns:
        latent: Spatial tensor [B, C, H, W]
    """
    B, HW, C = flattened.shape
    H, W = shape
    assert HW == H * W, f"Shape mismatch: {HW} != {H * W}"
    
    latent = flattened.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
    return latent


def prepare_latent_image_ids(batch_size: int, height: int, width: int, device='cpu', dtype=torch.float32) -> Tensor:
    """
    Prepare positional IDs for latent image patches.
    Based on flow repository implementation.
    
    Args:
        batch_size: Batch size
        height: Latent height (H//8)
        width: Latent width (W//8) 
        device: Device for tensor
        dtype: Data type for tensor
    Returns:
        image_ids: Position IDs [B, H*W, 3] for transformer
    """
    # Create 2D position grid
    y_ids = torch.arange(height, device=device, dtype=dtype)
    x_ids = torch.arange(width, device=device, dtype=dtype)
    
    y_grid, x_grid = torch.meshgrid(y_ids, x_ids, indexing='ij')
    
    # Create position IDs [H, W, 3] with (t=0, y, x) format
    image_ids = torch.zeros((height, width, 3), device=device, dtype=dtype)
    image_ids[:, :, 1] = y_grid  # Y coordinates
    image_ids[:, :, 2] = x_grid  # X coordinates
    
    # Expand for batch and flatten
    image_ids = image_ids.flatten(0, 1)  # [H*W, 3]
    image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 3]
    
    return image_ids