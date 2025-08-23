import torch
import numpy as np
from PIL import Image
import modules.processing as processing
import modules.shared as shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
import modules.sd_samplers as sd_samplers
import modules.images as images
from backend import memory_management


class RadianceProcessing(StableDiffusionProcessingTxt2Img):
    """
    Processing class specifically for Chroma Radiance models.
    Handles the unique aspects of radiance generation including direct latent space operations.
    """
    
    def __init__(self, **kwargs):
        # Extract radiance-specific parameters
        self.radiance_guidance = kwargs.pop('radiance_guidance', 0.0)
        self.radiance_attn_padding = kwargs.pop('radiance_attn_padding', 1)
        self.nerf_tile_size = kwargs.pop('nerf_tile_size', None)
        
        super().__init__(**kwargs)
        
        # Mark this as a radiance processing instance
        self.is_radiance = True
        
        # Override some default settings for radiance
        if 'sampler_name' not in kwargs:
            self.sampler_name = "Euler"  # Default sampler for radiance
        
    def setup_radiance_model(self):
        """Ensure we're using a radiance model."""
        if not (hasattr(shared.sd_model, 'is_radiance_model') and shared.sd_model.is_radiance_model()):
            raise RuntimeError("Current model is not a radiance model. Please load a radiance model first.")
        return True
        
    def get_conditions_for_radiance(self, prompts, steps):
        """Get text conditioning for radiance models."""
        self.setup_radiance_model()
        
        # Use the radiance model's text processing
        with torch.no_grad():
            cond = shared.sd_model.get_learned_conditioning(prompts)
            if isinstance(cond, dict):
                # Handle structured conditioning
                return cond
            else:
                # Handle simple tensor conditioning
                return {"context": cond}
    
    def create_radiance_latent(self, width, height, batch_size):
        """Create initial noise for radiance generation."""
        # Radiance models work directly in RGB space with proper scaling
        shape = (batch_size, 3, height, width)
        
        # Create noise in the appropriate range for radiance models
        latent = torch.randn(shape, device=shared.device, dtype=shared.sd_model.dtype)
        
        return latent
        
    def decode_radiance_latent(self, latent):
        """Convert radiance model output to displayable images."""
        # Radiance models output directly in image space
        with torch.no_grad():
            decoded = shared.sd_model.decode_first_stage(latent)
            
            # Ensure proper range and format
            decoded = torch.clamp(decoded, 0.0, 1.0)
            
            # Convert to PIL images
            images_list = []
            for i in range(decoded.shape[0]):
                img_tensor = decoded[i]
                # Convert from CHW to HWC and to numpy
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Convert to 0-255 range
                img_np = (img_np * 255).astype(np.uint8)
                # Create PIL image
                img_pil = Image.fromarray(img_np)
                images_list.append(img_pil)
                
        return images_list
        
    def sample_radiance(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        """Sample from radiance model."""
        self.setup_radiance_model()
        
        # Load model to GPU
        memory_management.load_model_gpu(shared.sd_model.forge_objects.unet.patcher)
        memory_management.load_model_gpu(shared.sd_model.forge_objects.clip.patcher)
        
        # Create initial latent
        x = self.create_radiance_latent(self.width, self.height, len(prompts))
        
        # Get sampler
        sampler = sd_samplers.create_sampler(self.sampler_name, shared.sd_model)
        
        # Prepare extra parameters for radiance model
        extra_params = {
            'radiance_guidance': self.radiance_guidance,
            'attn_padding': self.radiance_attn_padding,
        }
        
        # Sample using the radiance model
        with torch.no_grad():
            samples = sampler.sample(
                p=self,
                x=x,
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                steps=self.steps,
                image_conditioning=None,
                **extra_params
            )
        
        return samples
        
    def process_radiance_images(self):
        """Main processing function for radiance generation."""
        try:
            self.setup_radiance_model()
            
            # Get conditioning
            prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
            negative_prompts = self.negative_prompt if isinstance(self.negative_prompt, list) else [self.negative_prompt]
            
            # Expand prompts to match batch size
            while len(prompts) < self.batch_size:
                prompts.extend(prompts[:self.batch_size - len(prompts)])
            while len(negative_prompts) < self.batch_size:
                negative_prompts.extend(negative_prompts[:self.batch_size - len(negative_prompts)])
                
            prompts = prompts[:self.batch_size]
            negative_prompts = negative_prompts[:self.batch_size]
            
            # Get text conditioning
            conditioning = self.get_conditions_for_radiance(prompts, self.steps)
            unconditional_conditioning = self.get_conditions_for_radiance(negative_prompts, self.steps)
            
            # Generate seeds if needed
            if self.seed == -1:
                self.seed = int(torch.randint(0, 2**32, (1,)).item())
            
            seeds = [self.seed + i for i in range(self.batch_size)]
            subseeds = [0] * self.batch_size  # Not used in radiance models
            
            # Sample from model
            samples = self.sample_radiance(
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=0.0,
                prompts=prompts
            )
            
            # Decode to images
            result_images = self.decode_radiance_latent(samples)
            
            return result_images
            
        except Exception as e:
            print(f"Error in radiance processing: {e}")
            raise e
            
    def process_images_radiance(self):
        """Process images using radiance model - main entry point."""
        
        # Store original values
        original_prompt = self.prompt
        original_negative_prompt = self.negative_prompt
        
        results = []
        all_prompts = []
        all_negative_prompts = []
        all_seeds = []
        
        try:
            for i in range(self.n_iter):
                # Process batch
                batch_results = self.process_radiance_images()
                results.extend(batch_results)
                
                # Track metadata
                prompts_for_batch = [self.prompt] * self.batch_size if isinstance(self.prompt, str) else self.prompt[:self.batch_size]
                negative_prompts_for_batch = [self.negative_prompt] * self.batch_size if isinstance(self.negative_prompt, str) else self.negative_prompt[:self.batch_size]
                seeds_for_batch = [self.seed + i * self.batch_size + j for j in range(self.batch_size)]
                
                all_prompts.extend(prompts_for_batch)
                all_negative_prompts.extend(negative_prompts_for_batch)
                all_seeds.extend(seeds_for_batch)
                
                # Update seed for next iteration
                self.seed += self.batch_size
                
        finally:
            # Restore original values
            self.prompt = original_prompt  
            self.negative_prompt = original_negative_prompt
            
        # Create processed result
        processed = processing.Processed(
            p=self,
            images_list=results,
            seed=all_seeds[0] if all_seeds else self.seed,
            info="Radiance generation completed",
            subseed=0,
            all_prompts=all_prompts,
            all_negative_prompts=all_negative_prompts,
            all_seeds=all_seeds,
            all_subseeds=[0] * len(all_seeds),
            index_of_first_image=0,
        )
        
        return processed


def process_images_radiance(p: RadianceProcessing):
    """Process images using radiance model - external interface."""
    return p.process_images_radiance()


def create_radiance_processing(**kwargs):
    """Factory function to create RadianceProcessing instances."""
    return RadianceProcessing(**kwargs)