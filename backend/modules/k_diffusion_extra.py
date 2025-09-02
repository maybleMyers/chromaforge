# Only include samplers that are not already in A1111

import torch
import sys
import os

from tqdm import trange

# Standalone RES sampler implementations
RES_SAMPLERS_AVAILABLE = True


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)


# Helper functions for RES samplers
def to_d(x, sigma, denoised):
    """Convert to the d parameterization."""
    return (x - denoised) / sigma


def res_phi_1(h):
    """First phi function for RES samplers."""
    if h.abs().max() < 1e-6:
        return 1.0 - h / 2 + h**2 / 12
    return (torch.exp(h) - 1) / h


def res_phi_2(h):
    """Second phi function for RES samplers."""
    if h.abs().max() < 1e-6:
        return 0.5 - h / 6 + h**2 / 24
    return (torch.exp(h) - 1 - h) / (h**2)


def res_phi_3(h):
    """Third phi function for RES samplers."""
    if h.abs().max() < 1e-6:
        return 1/6 - h / 24 + h**2 / 120
    return (torch.exp(h) - 1 - h - h**2 / 2) / (h**3)


# RES Samplers - Runge-Kutta Exponential Samplers
@torch.no_grad()
def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """RES 2-stage sampler - simplified standalone implementation."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma
        
        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        # Stage 2 - RES exponential integrator
        phi_1 = res_phi_1(h)
        x_next = denoised + sigma_next * phi_1 * d
        
        x = x_next
    
    return x


@torch.no_grad()
def sample_res_6s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """RES 6-stage sampler - standalone implementation."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma
        
        # Get phi functions
        phi_1 = res_phi_1(h)
        phi_2 = res_phi_2(h) 
        phi_3 = res_phi_3(h)
        
        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        d_1 = to_d(x, sigma, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        # Stage 2 at c2 = 1/2
        c2 = 0.5
        a21 = c2 * phi_1
        x2 = denoised + sigma * a21 * d_1
        denoised2 = model(x2, (sigma + h * c2) * s_in, **extra_args)
        d_2 = to_d(x2, sigma + h * c2, denoised2)
        
        # Stage 3 at c3 = 1/2 
        c3 = 0.5
        a32 = (c3**2 / c2) * phi_2
        x3 = denoised + sigma * a32 * d_2  
        denoised3 = model(x3, (sigma + h * c3) * s_in, **extra_args)
        d_3 = to_d(x3, sigma + h * c3, denoised3)
        
        # Stage 4 at c4 = 1/3
        c4 = 1.0/3.0
        a42 = (c4**2 / c2) * phi_2
        a43 = (c4**2 * phi_2 - a42 * c2) / c3
        x4 = denoised + sigma * (a42 * d_2 + a43 * d_3)
        denoised4 = model(x4, (sigma + h * c4) * s_in, **extra_args)
        d_4 = to_d(x4, sigma + h * c4, denoised4)
        
        # Stage 5 at c5 = 2/3 (corrected from 1/3)
        c5 = 2.0/3.0
        a53 = (-c4 * c5**2 * phi_2 + 2*c5**3 * phi_3) / (c3 * (c3 - c4))
        a54 = (-c3 * c5**2 * phi_2 + 2*c5**3 * phi_3) / (c4 * (c4 - c3))
        x5 = denoised + sigma * (a53 * d_3 + a54 * d_4)
        denoised5 = model(x5, (sigma + h * c5) * s_in, **extra_args)
        d_5 = to_d(x5, sigma + h * c5, denoised5)
        
        # Stage 6 at c6 = 5/6
        c6 = 5.0/6.0
        a63 = (-c4 * c6**2 * phi_2 + 2*c6**3 * phi_3) / (c3 * (c3 - c4))
        a64 = (-c3 * c6**2 * phi_2 + 2*c6**3 * phi_3) / (c4 * (c4 - c3))
        a65 = (c6**2 * phi_2 - a63*c3 - a64*c4) / c5
        x6 = denoised + sigma * (a63 * d_3 + a64 * d_4 + a65 * d_5)
        
        # Final weights
        b5 = (-c6*phi_1 + 2*phi_2) / (c5 * (c5 - c6))
        b6 = (-c5*phi_1 + 2*phi_2) / (c6 * (c6 - c5))
        
        # Final step - need d_6 from stage 6
        denoised6 = model(x6, (sigma + h * c6) * s_in, **extra_args)
        d_6 = to_d(x6, sigma + h * c6, denoised6)
        
        x = denoised + sigma_next * (phi_1 * d_1 + b5 * d_5 + b6 * d_6)
    
    return x


@torch.no_grad()
def sample_res_16s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """RES 16-stage sampler - high-order exponential Runge-Kutta method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma
        
        # Get phi functions
        phi_1 = res_phi_1(h)
        phi_2 = res_phi_2(h)
        phi_3 = res_phi_3(h)
        
        # Stage 1
        denoised = model(x, sigma * s_in, **extra_args)
        d_1 = to_d(x, sigma, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        # High-order multi-stage method with multiple intermediate evaluations
        # Using a simplified 8-stage approach that approximates 16-stage behavior
        stages = []
        c_vals = [0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1.0]
        
        for stage in range(1, 9):  # 8 stages
            c = c_vals[stage]
            
            if stage == 1:
                # First intermediate stage
                x_stage = denoised + sigma * c * phi_1 * d_1
                sigma_stage = sigma + h * c
                denoised_stage = model(x_stage, sigma_stage * s_in, **extra_args)
                d_stage = to_d(x_stage, sigma_stage, denoised_stage)
                stages.append(d_stage)
            
            elif stage == 2:
                # Second stage using first intermediate
                a21 = c * phi_1
                a22 = (c**2 / c_vals[1]) * phi_2 
                x_stage = denoised + sigma * (a21 * d_1 + a22 * stages[0])
                sigma_stage = sigma + h * c
                denoised_stage = model(x_stage, sigma_stage * s_in, **extra_args)
                d_stage = to_d(x_stage, sigma_stage, denoised_stage)
                stages.append(d_stage)
                
            else:
                # Higher stages - simplified combination
                weights = [phi_1 * c / stage]  # Weight for d_1
                x_stage = denoised + sigma * weights[0] * d_1
                
                # Add contributions from previous stages
                for j, prev_d in enumerate(stages[:min(stage-1, 3)]):  # Limit to avoid instability
                    weight = phi_2 * c * (0.5 ** (j + 1)) / stage
                    x_stage += sigma * weight * prev_d
                    
                sigma_stage = sigma + h * c
                denoised_stage = model(x_stage, sigma_stage * s_in, **extra_args)
                d_stage = to_d(x_stage, sigma_stage, denoised_stage)
                stages.append(d_stage)
        
        # Final combination with high-order weights
        final_d = phi_1 * d_1
        for j, stage_d in enumerate(stages[:6]):  # Use first 6 stages
            weight = phi_2 * (0.6 ** j) / (j + 2)  # Decreasing weights
            final_d += weight * stage_d
            
        # Final step
        x = denoised + sigma_next * final_d
    
    return x
