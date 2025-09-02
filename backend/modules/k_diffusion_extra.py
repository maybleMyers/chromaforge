# Only include samplers that are not already in A1111

import torch
import sys
import os
import math
from itertools import permutations, combinations

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


# RES4LYF phi functions - copied from working implementation
def _gamma(n: int) -> int:
    """Gamma function for positive integers: Γ(n) = (n-1)!"""
    return math.factorial(n-1)

def _incomplete_gamma(s: int, x: float, gamma_s=None) -> float:
    """Incomplete gamma function for positive integer s"""
    if gamma_s is None:
        gamma_s = _gamma(s)
    sum_ = 0.0
    for k in range(s):
        sum_ += (x**k) / math.factorial(k)
    return sum_ * math.exp(-x) * gamma_s

def phi(j: int, neg_h: float):
    """RES4LYF phi function implementation"""
    assert j > 0
    gamma_ = _gamma(j)
    incomp_gamma_ = _incomplete_gamma(j, neg_h, gamma_s=gamma_)
    phi_ = math.exp(neg_h) * (neg_h**-j) * (1 - incomp_gamma_/gamma_)
    return phi_

# Additional phi functions for higher orders
def res_phi_4(h):
    """Fourth phi function for RES samplers."""
    if h.abs().max() < 1e-6:
        return 1/24 - h / 120 + h**2 / 720
    return (torch.exp(h) - 1 - h - h**2 / 2 - h**3 / 6) / (h**4)

def res_phi_5(h):
    """Fifth phi function for RES samplers."""  
    if h.abs().max() < 1e-6:
        return 1/120 - h / 720 + h**2 / 5040
    return (torch.exp(h) - 1 - h - h**2 / 2 - h**3 / 6 - h**4 / 24) / (h**5)

def res_phi_6(h):
    """Sixth phi function for RES samplers."""
    if h.abs().max() < 1e-6:
        return 1/720 - h / 5040 + h**2 / 40320
    return (torch.exp(h) - 1 - h - h**2 / 2 - h**3 / 6 - h**4 / 24 - h**5 / 120) / (h**6)

class Phi:
    """RES4LYF Phi class - copied from working implementation"""
    def __init__(self, h, c, analytic_solution=False): 
        self.h = h
        self.c = c
        self.cache = {}
        self.phi_f = phi

    def __call__(self, j, i=-1):
        if (j, i) in self.cache:
            return self.cache[(j, i)]

        if i < 0:
            c = 1
        else:
            c = self.c[i - 1]
            if c == 0:
                self.cache[(j, i)] = 0
                return 0

        if j == 0:
            result = math.exp(float(-self.h * c))
        else:
            result = self.phi_f(j, -self.h * c)

        self.cache[(j, i)] = result
        return result

# Legacy phi functions for backward compatibility
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


# RES4LYF helper functions - EXACT copies
def theta_numerator(j, cd, ci, ck, cj, cl):
    if j == 2:
        numerator = -cj * cd * ck * cl
    if j == 3:
        numerator = 2 * (cj * ck * cd + cj*ck*cl + ck*cd*cl + cd*cl*cj)
    if j == 4:
        numerator = -6*(cj*ck + cj*cd + cj*cl + ck*cd + ck*cl + cd*cl)
    if j == 5:
        numerator = 24 * (cj + ck + cl + cd)
    if j == 6:
        numerator = -120
    return numerator

def theta(j, cd, ci, ck, cj, cl):
    if j == 2:
        numerator = -cj * cd * ck * cl
    if j == 3:
        numerator = 2 * (cj * ck * cd + cj*ck*cl + ck*cd*cl + cd*cl*cj)
    if j == 4:
        numerator = -6*(cj*ck + cj*cd + cj*cl + ck*cd + ck*cl + cd*cl)
    if j == 5:
        numerator = 24 * (cj + ck + cl + cd)
    if j == 6:
        numerator = -120
    return numerator / (ci * (ci - cj) * (ci - ck) * (ci - cl) * (ci - cd))

def prod_diff(cj, ck, cl=None, cd=None):
    if cl is None and cd is None:
        return cj * (cj - ck)
    if cd is None:
        return cj * (cj - ck) * (cj - cl)
    else:
        return cj * (cj - ck) * (cj - cl) * (cj - cd)

def denominator(ci, *args):
    result = ci 
    for arg in args:
        result *= (ci - arg)
    return result

def get_res_6s_coefficients(h):
    """Get RES 6s coefficients - copied exactly from RES4LYF"""
    # Original c-values from RES4LYF (with division by zero issue)
    c1, c2, c3, c4, c5, c6 = 0, 1/2, 1/2, 1/3, 1/3, 5/6
    ci = [c1, c2, c3, c4, c5, c6]
    φ = Phi(h, ci, analytic_solution=False)
    
    # Coefficient calculation - exact copy from RES4LYF
    a2_1 = c2 * φ(1,2)
    
    a3_1 = 0
    a3_2 = (c3**2 / c2) * φ(2,3)
    
    a4_1 = 0
    a4_2 = (c4**2 / c2) * φ(2,4)
    a4_3 = (c4**2 * φ(2,4) - a4_2 * c2) / c3
    
    a5_1 = 0
    a5_2 = 0 #zero
    # Handle division by zero - use L'Hôpital's rule limit or special case
    if abs(c3 - c4) < 1e-10:  # c3 == c4 case
        # Use limit as c3 -> c4
        a5_3 = 0  # This is what the limit evaluates to
        a5_4 = 0
    else:
        a5_3 = (-c4 * c5**2 * φ(2,5) + 2*c5**3 * φ(3,5)) / (c3 * (c3 - c4))
        a5_4 = (-c3 * c5**2 * φ(2,5) + 2*c5**3 * φ(3,5)) / (c4 * (c4 - c3))
    
    a6_1 = 0
    a6_2 = 0 #zero
    if abs(c3 - c4) < 1e-10:  # c3 == c4 case
        a6_3 = 0
        a6_4 = 0
    else:
        a6_3 = (-c4 * c6**2 * φ(2,6) + 2*c6**3 * φ(3,6)) / (c3 * (c3 - c4))
        a6_4 = (-c3 * c6**2 * φ(2,6) + 2*c6**3 * φ(3,6)) / (c4 * (c4 - c3))
    a6_5 = (c6**2 * φ(2,6) - a6_3*c3 - a6_4*c4) / c5
            
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = (-c6*φ(2) + 2*φ(3)) / (c5 * (c5 - c6))
    b6 = (-c5*φ(2) + 2*φ(3)) / (c6 * (c6 - c5))

    a = [
        [0, 0, 0, 0, 0, 0],
        [a2_1, 0, 0, 0, 0, 0],  # First column from gen_first_col_exp
        [0, a3_2, 0, 0, 0, 0],
        [0, a4_2, a4_3, 0, 0, 0],
        [0, a5_2, a5_3, a5_4, 0, 0],
        [0, a6_2, a6_3, a6_4, a6_5, 0],
    ]
    b = [b1, b2, b3, b4, b5, b6]
    
    return a, b, ci


def get_res_16s_coefficients(h):
    """Get RES 16s coefficients - EXACT copy from RES4LYF"""
    use_analytic_solution = False  # Use same as RES4LYF default
    
    c1 = 0
    c2 = c3 = c5 = c8 = c12 = 1/2
    c4 = c11 = c15 = 1/3
    c6 = c9 = c13 = 1/5
    c7 = c10 = c14 = 1/4
    c16 = 1
    ci = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
    φ = Phi(h, ci, analytic_solution=use_analytic_solution)
    
    a3_2 = (1/2) * φ(2,3)

    a = [[0.0 for _ in range(16)] for _ in range(16)]
    b = [[0.0 for _ in range(16)]]

    for i in range(3, 5): # i=3,4     j=2
        j=2
        a[i-1][j-1] = (ci[i-1]**2 / ci[j-1]) * φ(j,i)
    
    for i in range(5, 8): # i=5,6,7   j,k ∈ {3, 4}, j != k
        jk = list(permutations([3, 4], 2)) 
        for j,k in jk:
            a[i-1][j-1] = (-ci[i-1]**2 * ci[k-1] * φ(2,i) + 2*ci[i-1]**3 * φ(3,i)) / prod_diff(ci[j-1], ci[k-1])
            
    for i in range(8, 12): # i=8,9,10,11  j,k,l ∈ {5, 6, 7}, j != k != l
        jkl = list(permutations([5, 6, 7], 3)) 
        for j,k,l in jkl:
            a[i-1][j-1] = (ci[i-1]**2 * ci[k-1] * ci[l-1] * φ(2,i) - 2*ci[i-1]**3 * (ci[k-1] + ci[l-1]) * φ(3,i) + 6*ci[i-1]**4 * φ(4,i)) / (ci[j-1] * (ci[j-1] - ci[k-1]) * (ci[j-1] - ci[l-1]))

    for i in range(12,16): # i=12,13,14,15
        jkld = list(permutations([8,9,10,11], 4)) 
        for j,k,l,d in jkld:
            numerator = -ci[i-1]**2 * ci[d-1]*ci[k-1]*ci[l-1] * φ(2,i) + 2*ci[i-1]**3 * (ci[d-1]*ci[k-1] + ci[d-1]*ci[l-1] + ci[k-1]*ci[l-1]) * φ(3,i) - 6*ci[i-1]**4 * (ci[d-1] + ci[k-1] + ci[l-1]) * φ(4,i) + 24*ci[i-1]**5 * φ(5,i)
            a[i-1][j-1] = numerator / prod_diff(ci[j-1], ci[k-1], ci[l-1], ci[d-1])
            
    ijdkl = list(permutations([12,13,14,15,16], 5)) 
    for i,j,d,k,l in ijdkl:
        b[0][i-1] = theta(2, ci[d-1], ci[i-1], ci[k-1], ci[j-1], ci[l-1]) * φ(2) + theta(3, ci[d-1], ci[i-1], ci[k-1], ci[j-1], ci[l-1])*φ(3) + theta(4, ci[d-1], ci[i-1], ci[k-1], ci[j-1], ci[l-1])*φ(4) + theta(5, ci[d-1], ci[i-1], ci[k-1], ci[j-1], ci[l-1])*φ(5) + theta(6, ci[d-1], ci[i-1], ci[k-1], ci[j-1], ci[l-1]) * φ(6)

    return a, b, ci


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
    """RES 6-stage sampler - exact copy of RES4LYF math."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = float(sigma_next - sigma)
        
        # Get coefficients using exact RES4LYF calculation
        a, b, ci = get_res_6s_coefficients(h)
        
        # Convert to proper format
        num_stages = len(ci)
        
        # Stage computations - exact RK method
        k = []  # Stage derivatives
        
        for stage in range(num_stages):
            if stage == 0:
                # First stage at current point
                denoised = model(x, sigma * s_in, **extra_args)
                k_i = to_d(x, sigma, denoised)
                
                if callback is not None:
                    callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
            else:
                # Intermediate stages
                x_stage = x
                for j in range(stage):
                    x_stage = x_stage + h * a[stage][j] * k[j]
                
                sigma_stage = sigma + h * ci[stage]
                denoised_stage = model(x_stage, sigma_stage * s_in, **extra_args)
                k_i = to_d(x_stage, sigma_stage, denoised_stage)
            
            k.append(k_i)
        
        # Final integration step using RK formula: x_new = x + h * sum(b_i * k_i)
        x_new = x
        for j in range(num_stages):
            x_new = x_new + h * b[j] * k[j]
            
        x = x_new
    
    return x


@torch.no_grad()
def sample_res_16s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """RES 16-stage sampler - EXACT copy of RES4LYF math."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = float(sigma_next - sigma)
        
        # Get coefficients using EXACT RES4LYF calculation
        a, b, ci = get_res_16s_coefficients(h)
        
        # Convert to proper format
        num_stages = len(ci)
        
        # Stage computations - exact RK method
        k = []  # Stage derivatives
        
        for stage in range(num_stages):
            if stage == 0:
                # First stage at current point
                denoised = model(x, sigma * s_in, **extra_args)
                k_i = to_d(x, sigma, denoised)
                
                if callback is not None:
                    callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
            else:
                # Intermediate stages
                x_stage = x
                for j in range(stage):
                    x_stage = x_stage + h * a[stage][j] * k[j]
                
                sigma_stage = sigma + h * ci[stage]
                denoised_stage = model(x_stage, sigma_stage * s_in, **extra_args)
                k_i = to_d(x_stage, sigma_stage, denoised_stage)
            
            k.append(k_i)
        
        # Final integration step using RK formula: x_new = x + h * sum(b_i * k_i)
        x_new = x
        for j in range(num_stages):
            x_new = x_new + h * b[0][j] * k[j]  # Note: b is nested list for RES 16s
            
        x = x_new
    
    return x
