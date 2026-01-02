"""
Advanced Exponential Integrator Samplers

Provides RES 2M (multistep), RES 2S (single-step), and ER-SDE exponential integrator
samplers. These samplers are particularly well-suited for flow-matching models like
Z-Image DCT.

RES 2M: 2-step multistep exponential integrator
RES 2S: 2-stage single-step exponential integrator (midpoint method)
ER-SDE: Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3) from arXiv:2309.06169
"""

import torch
from tqdm.auto import trange
import math


def phi_1(h):
    """First-order phi function: (exp(h) - 1) / h"""
    return h.expm1() / h if abs(h.item()) > 1e-4 else 1.0 + h / 2


def phi_2(h):
    """Second-order phi function: (exp(h) - 1 - h) / h^2"""
    if abs(h.item()) > 1e-4:
        return (h.expm1() - h) / (h * h)
    return 0.5 + h / 6


@torch.no_grad()
def sample_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.0, s_noise=1.0):
    """
    RES 2M: 2-step multistep exponential integrator sampler.

    Uses the previous denoised prediction to improve accuracy.
    Similar to DPM++ 2M but with exponential integrator formulation.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    t_fn = lambda sigma: sigma.log().neg()
    sigma_fn = lambda t: t.neg().exp()

    old_denoised = None
    old_sigma = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        if sigma_next == 0:
            x = denoised
        else:
            t = t_fn(sigma)
            t_next = t_fn(sigma_next)
            h = t_next - t

            if old_denoised is None:
                x = sigma_fn(t_next) / sigma_fn(t) * x - (-h).expm1() * denoised
            else:
                t_old = t_fn(old_sigma)
                h_prev = t - t_old

                r = h_prev / h

                b1 = phi_1(-h)
                b2 = phi_2(-h) / r if abs(r) > 1e-6 else 0.0

                eps_curr = denoised - x
                eps_prev = old_denoised - x

                h_sum = h * (b1 * eps_curr + b2 * (eps_curr - eps_prev))

                x = torch.exp(-h) * x + h_sum

        old_denoised = denoised
        old_sigma = sigma

    return x


@torch.no_grad()
def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.0, s_noise=1.0, c2=0.5):
    """
    RES 2S: 2-stage single-step exponential integrator sampler.

    Uses a midpoint evaluation for second-order accuracy.
    c2 controls the midpoint position (default 0.5).
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    t_fn = lambda sigma: sigma.log().neg()
    sigma_fn = lambda t: t.neg().exp()

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        if sigma_next == 0:
            x = denoised
        else:
            t = t_fn(sigma)
            t_next = t_fn(sigma_next)
            h = t_next - t

            t_mid = t + c2 * h
            sigma_mid = sigma_fn(t_mid)

            x_mid = sigma_fn(t_mid) / sigma_fn(t) * x - (-(c2 * h)).expm1() * denoised

            denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)

            phi1 = phi_1(-h)
            phi2 = phi_2(-h)

            b1 = phi1 - phi2 / c2
            b2 = phi2 / c2

            eps_1 = denoised - x
            eps_2 = denoised_mid - x

            h_sum = h * (b1 * eps_1 + b2 * eps_2)

            x = torch.exp(-h) * x + h_sum

    return x


@torch.no_grad()
def sample_res_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.5, s_noise=1.0, noise_sampler=None):
    """
    RES 2M SDE: Stochastic version of RES 2M with noise injection.
    """
    from k_diffusion.sampling import BrownianTreeNoiseSampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)

    t_fn = lambda sigma: sigma.log().neg()
    sigma_fn = lambda t: t.neg().exp()

    old_denoised = None
    old_sigma = None

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        if sigma_next == 0:
            x = denoised
        else:
            t = t_fn(sigma)
            t_next = t_fn(sigma_next)
            h = t_next - t
            eta_h = eta * h

            if old_denoised is None:
                x = sigma_fn(t_next) / sigma_fn(t) * (-eta_h).exp() * x - (-h - eta_h).expm1() * denoised
            else:
                t_old = t_fn(old_sigma)
                h_prev = t - t_old
                r = h_prev / h

                b2 = phi_2(-h - eta_h) / r if abs(r) > 1e-6 else 0.0

                denoised_d = denoised + b2 * (denoised - old_denoised)

                x = sigma_fn(t_next) / sigma_fn(t) * (-eta_h).exp() * x - (-h - eta_h).expm1() * denoised_d

            if eta > 0:
                noise = noise_sampler(sigma, sigma_next)
                x = x + noise * sigma_next * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        old_sigma = sigma

    return x


@torch.no_grad()
def sample_res_2s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=0.5, s_noise=1.0, c2=0.5, noise_sampler=None):
    """
    RES 2S SDE: Stochastic version of RES 2S with noise injection.
    """
    from k_diffusion.sampling import BrownianTreeNoiseSampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)

    t_fn = lambda sigma: sigma.log().neg()
    sigma_fn = lambda t: t.neg().exp()

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        if sigma_next == 0:
            x = denoised
        else:
            t = t_fn(sigma)
            t_next = t_fn(sigma_next)
            h = t_next - t
            eta_h = eta * h

            t_mid = t + c2 * h
            sigma_mid = sigma_fn(t_mid)

            x_mid = sigma_fn(t_mid) / sigma_fn(t) * x - (-(c2 * h)).expm1() * denoised

            denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)

            phi1 = phi_1(-h - eta_h)
            phi2 = phi_2(-h - eta_h)

            b1 = phi1 - phi2 / c2
            b2 = phi2 / c2

            eps_1 = denoised - x
            eps_2 = denoised_mid - x

            h_sum = h * (b1 * eps_1 + b2 * eps_2)

            x = torch.exp(-h - eta_h) * x + h_sum

            if eta > 0:
                noise = noise_sampler(sigma, sigma_next)
                x = x + noise * sigma_next * (-2 * eta_h).expm1().neg().sqrt() * s_noise

    return x


def _default_noise_sampler(x, seed=None):
    """Create a default noise sampler for ER-SDE."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device).manual_seed(seed)

    def sampler(sigma, sigma_next):
        return torch.randn_like(x, generator=generator)

    return sampler


def _sigma_to_half_log_snr(sigma, model_sampling=None):
    """Convert sigma to half log SNR (lambda)."""
    # For VP models: lambda = -0.5 * log(sigma^2 / (1 + sigma^2))
    # Simplified: lambda = -0.5 * log(sigma^2) for large sigma
    return -0.5 * torch.log(sigma ** 2 + 1e-8)


def _offset_first_sigma_for_snr(sigmas, model_sampling=None):
    """Offset first sigma to avoid numerical issues at high SNR."""
    # Small offset to prevent division by zero in lambda space
    sigmas = sigmas.clone()
    if sigmas[0] < 1e-6:
        sigmas[0] = 1e-6
    return sigmas


@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  s_noise=1.0, noise_sampler=None, max_stage=3):
    """
    Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3).

    From arXiv:2309.06169 "ER-SDE-Solver: A Third-Order SDE Solver for
    VP-Type Diffusion ODEs and SDEs with Exponential Noise Schedule"

    This is a multi-stage solver that uses lambda (half log SNR) space
    for integration, providing improved accuracy for VP-type diffusion.

    Args:
        model: The denoising model
        x: Initial latent
        sigmas: Noise schedule
        extra_args: Extra arguments for the model
        callback: Callback function for progress
        disable: Disable progress bar
        s_noise: Noise multiplier
        noise_sampler: Custom noise sampler
        max_stage: Maximum number of stages (1, 2, or 3)

    Returns:
        Denoised latent
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    if noise_sampler is None:
        noise_sampler = _default_noise_sampler(x, seed=seed)

    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(lam):
        """Default noise scaling function for ER-SDE."""
        return lam * ((lam ** 0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler
    num_integration_points = 200.0
    point_indices = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    # Get model sampling parameters if available
    model_sampling = None
    try:
        if hasattr(model, 'inner_model') and hasattr(model.inner_model, 'model_patcher'):
            model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    except:
        pass

    sigmas = _offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = _sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        stage_used = min(max_stage, i + 1)

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s = er_lambdas[i]
            er_lambda_t = er_lambdas[i + 1]

            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1: Euler step
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            # Stage 2: First-order correction
            if stage_used >= 2 and old_denoised is not None:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indices * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)
                s = torch.sum(1 / scaled_pos) * lambda_step_size

                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1] + 1e-8)
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                # Stage 3: Second-order correction
                if stage_used >= 3 and old_denoised_d is not None and i >= 2:
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2 + 1e-8)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u

                old_denoised_d = denoised_d

            # Add stochastic noise
            if s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                noise_scale = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt()
                noise_scale = noise_scale.nan_to_num(nan=0.0)
                x = x + alpha_t * noise * s_noise * noise_scale

        old_denoised = denoised

    return x
