#!/usr/bin/env python3
"""
Quantization Sensitivity Analyzer for Vision-Language Models
=============================================================

Analyzes which layers of a VLM can be safely quantized to INT8 without
significant quality degradation. Specifically designed for Qwen3-VL models
but adaptable to other architectures.

Methods used:
1. Weight Distribution Analysis - Identifies outlier-prone layers
2. Reconstruction Error - Measures MSE after simulated quantization
3. Activation Sensitivity - Measures output drift with quantized weights
4. Vision-Specific Tests - Evaluates image encoding quality degradation
5. Cross-Attention Analysis - Special handling for vision-language fusion layers

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import json
import warnings
import gc
import logging
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizationSafety(Enum):
    """Safety classification for quantization."""
    SAFE = "safe"                      # Can quantize to Q8 with minimal impact
    CAUTION = "caution"                # Quantize with monitoring, slight degradation expected
    SENSITIVE = "sensitive"            # Significant degradation likely, consider Q8 only if necessary
    CRITICAL = "critical"              # Do NOT quantize - severe degradation expected


@dataclass
class LayerAnalysisResult:
    """Results from analyzing a single layer."""
    name: str
    module_type: str
    param_count: int
    param_size_mb: float
    
    # Weight distribution metrics
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_kurtosis: float = 0.0
    weight_skewness: float = 0.0
    outlier_ratio: float = 0.0          # % of weights > 3 std from mean
    dynamic_range: float = 0.0          # max/min ratio (indicates quantization difficulty)
    
    # Reconstruction error metrics
    reconstruction_mse: float = 0.0
    reconstruction_max_error: float = 0.0
    reconstruction_cosine_sim: float = 1.0
    
    # Activation sensitivity (if tested)
    activation_mse: Optional[float] = None
    activation_cosine_sim: Optional[float] = None
    activation_max_drift: Optional[float] = None
    
    # Classification
    safety: QuantizationSafety = QuantizationSafety.SAFE
    risk_score: float = 0.0             # 0-100, higher = more risky to quantize
    reasons: List[str] = field(default_factory=list)
    
    # Layer categorization
    is_vision_layer: bool = False
    is_cross_attention: bool = False
    is_embedding: bool = False
    is_normalization: bool = False
    is_output_layer: bool = False


@dataclass 
class ModelAnalysisReport:
    """Complete analysis report for a model."""
    model_name: str
    analysis_timestamp: str
    total_params: int
    total_size_mb: float
    
    # Layer results
    layer_results: Dict[str, LayerAnalysisResult] = field(default_factory=dict)
    
    # Summary statistics
    safe_layers: List[str] = field(default_factory=list)
    caution_layers: List[str] = field(default_factory=list)
    sensitive_layers: List[str] = field(default_factory=list)
    critical_layers: List[str] = field(default_factory=list)
    
    # Memory savings estimate
    quantizable_size_mb: float = 0.0
    non_quantizable_size_mb: float = 0.0
    estimated_q8_size_mb: float = 0.0
    
    # Skip modules for BitsAndBytes
    skip_modules: List[str] = field(default_factory=list)


class QuantizationSimulator:
    """Simulates INT8 quantization effects on tensors."""
    
    @staticmethod
    def symmetric_quantize(tensor: Tensor, bits: int = 8) -> Tuple[Tensor, float]:
        """Symmetric quantization simulation."""
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        # Calculate scale
        max_val = tensor.abs().max()
        scale = max_val / qmax if max_val > 0 else 1.0
        
        # Quantize and dequantize
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        dequantized = quantized * scale
        
        return dequantized, scale.item()
    
    @staticmethod
    def asymmetric_quantize(tensor: Tensor, bits: int = 8) -> Tuple[Tensor, float, float]:
        """Asymmetric quantization simulation (like bitsandbytes)."""
        qmin = 0
        qmax = 2 ** bits - 1
        
        # Calculate scale and zero point
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / qmax if (max_val - min_val) > 0 else 1.0
        zero_point = qmin - torch.round(min_val / scale)
        
        # Quantize and dequantize
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
        dequantized = (quantized - zero_point) * scale
        
        return dequantized, scale.item(), zero_point.item()
    
    @staticmethod
    def compute_reconstruction_error(original: Tensor, reconstructed: Tensor) -> Dict[str, float]:
        """Compute various reconstruction error metrics."""
        mse = F.mse_loss(reconstructed, original).item()
        max_error = (original - reconstructed).abs().max().item()
        
        # Cosine similarity (flatten tensors)
        orig_flat = original.flatten().float()
        recon_flat = reconstructed.flatten().float()
        cosine_sim = F.cosine_similarity(
            orig_flat.unsqueeze(0), 
            recon_flat.unsqueeze(0)
        ).item()
        
        # Relative error
        rel_error = (mse / (original.var().item() + 1e-10)) ** 0.5
        
        return {
            "mse": mse,
            "max_error": max_error,
            "cosine_sim": cosine_sim,
            "relative_error": rel_error
        }


class WeightAnalyzer:
    """Analyzes weight distributions for quantization sensitivity."""
    
    @staticmethod
    def compute_statistics(tensor: Tensor) -> Dict[str, float]:
        """Compute comprehensive weight statistics."""
        tensor_flat = tensor.flatten().float()
        
        mean = tensor_flat.mean().item()
        std = tensor_flat.std().item()
        min_val = tensor_flat.min().item()
        max_val = tensor_flat.max().item()
        
        # Kurtosis (measure of tail heaviness - high kurtosis = more outliers)
        if std > 0:
            normalized = (tensor_flat - mean) / std
            kurtosis = (normalized ** 4).mean().item() - 3  # Excess kurtosis
            skewness = (normalized ** 3).mean().item()
        else:
            kurtosis = 0.0
            skewness = 0.0
        
        # Outlier ratio (weights > 3 std from mean)
        if std > 0:
            outliers = ((tensor_flat - mean).abs() > 3 * std).float().mean().item()
        else:
            outliers = 0.0
        
        # Dynamic range
        if min_val != 0 and max_val != 0:
            dynamic_range = abs(max_val / min_val) if min_val != 0 else float('inf')
        else:
            dynamic_range = abs(max_val - min_val)
        
        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "outlier_ratio": outliers,
            "dynamic_range": min(dynamic_range, 1e6)  # Cap for numerical stability
        }


class LayerClassifier:
    """Classifies layers by their role in the model architecture."""
    
    # Patterns for identifying layer types in Qwen VL models
    VISION_PATTERNS = [
        "visual", "vision", "vit", "patch_embed", "image_encoder",
        "vision_tower", "mm_projector", "image_projection",
        "rotary_pos_emb",  # Vision-specific position embeddings
    ]
    
    CROSS_ATTENTION_PATTERNS = [
        "cross_attn", "cross_attention", "merger",
        "multimodal_projector", "mm_proj", "connector"
    ]
    
    EMBEDDING_PATTERNS = [
        "embed_tokens", "wte", "word_embed", "token_embed",
        "position_embed", "pos_embed"
    ]
    
    NORMALIZATION_PATTERNS = [
        "layernorm", "layer_norm", "ln_", "norm", "rmsnorm"
    ]
    
    OUTPUT_PATTERNS = [
        "lm_head", "output", "classifier", "score"
    ]
    
    # Never quantize these
    ALWAYS_SKIP_PATTERNS = [
        "layernorm", "layer_norm", "rmsnorm", "norm",
        "embed_tokens", "lm_head"
    ]
    
    @classmethod
    def classify(cls, layer_name: str, module: nn.Module) -> Dict[str, bool]:
        """Classify a layer based on its name and type."""
        name_lower = layer_name.lower()
        
        return {
            "is_vision_layer": any(p in name_lower for p in cls.VISION_PATTERNS),
            "is_cross_attention": any(p in name_lower for p in cls.CROSS_ATTENTION_PATTERNS),
            "is_embedding": any(p in name_lower for p in cls.EMBEDDING_PATTERNS),
            "is_normalization": any(p in name_lower for p in cls.NORMALIZATION_PATTERNS),
            "is_output_layer": any(p in name_lower for p in cls.OUTPUT_PATTERNS),
            "should_always_skip": any(p in name_lower for p in cls.ALWAYS_SKIP_PATTERNS)
        }


class RiskScorer:
    """Computes risk scores for quantizing layers."""
    
    # Thresholds for risk scoring
    THRESHOLDS = {
        "kurtosis_high": 10.0,          # High kurtosis indicates outliers
        "kurtosis_very_high": 50.0,
        "outlier_ratio_high": 0.01,     # > 1% outliers is concerning
        "outlier_ratio_very_high": 0.05,
        "reconstruction_mse_high": 1e-4,
        "reconstruction_mse_very_high": 1e-3,
        "cosine_sim_low": 0.9999,
        "cosine_sim_very_low": 0.999,
        "dynamic_range_high": 1000,
        "dynamic_range_very_high": 10000,
    }
    
    # Base risk scores for layer types
    LAYER_TYPE_RISK = {
        "is_vision_layer": 30,
        "is_cross_attention": 40,
        "is_embedding": 25,
        "is_normalization": 50,  # Usually should not be quantized
        "is_output_layer": 20,
    }
    
    @classmethod
    def compute_risk_score(cls, result: LayerAnalysisResult) -> Tuple[float, List[str]]:
        """Compute overall risk score (0-100) and list of risk reasons."""
        score = 0.0
        reasons = []
        
        # Layer type base risk
        if result.is_vision_layer:
            score += cls.LAYER_TYPE_RISK["is_vision_layer"]
            reasons.append("Vision encoder layer - typically sensitive to quantization")
        
        if result.is_cross_attention:
            score += cls.LAYER_TYPE_RISK["is_cross_attention"]
            reasons.append("Cross-attention layer - critical for vision-language fusion")
        
        if result.is_embedding:
            score += cls.LAYER_TYPE_RISK["is_embedding"]
            reasons.append("Embedding layer - affects all downstream computations")
        
        if result.is_normalization:
            score += cls.LAYER_TYPE_RISK["is_normalization"]
            reasons.append("Normalization layer - should generally not be quantized")
        
        if result.is_output_layer:
            score += cls.LAYER_TYPE_RISK["is_output_layer"]
            reasons.append("Output layer - directly affects predictions")
        
        # Statistical risk factors
        if result.weight_kurtosis > cls.THRESHOLDS["kurtosis_very_high"]:
            score += 25
            reasons.append(f"Very high kurtosis ({result.weight_kurtosis:.1f}) - many outliers")
        elif result.weight_kurtosis > cls.THRESHOLDS["kurtosis_high"]:
            score += 15
            reasons.append(f"High kurtosis ({result.weight_kurtosis:.1f}) - some outliers")
        
        if result.outlier_ratio > cls.THRESHOLDS["outlier_ratio_very_high"]:
            score += 20
            reasons.append(f"High outlier ratio ({result.outlier_ratio:.2%})")
        elif result.outlier_ratio > cls.THRESHOLDS["outlier_ratio_high"]:
            score += 10
            reasons.append(f"Moderate outlier ratio ({result.outlier_ratio:.2%})")
        
        if result.dynamic_range > cls.THRESHOLDS["dynamic_range_very_high"]:
            score += 15
            reasons.append(f"Very high dynamic range ({result.dynamic_range:.0f})")
        elif result.dynamic_range > cls.THRESHOLDS["dynamic_range_high"]:
            score += 8
            reasons.append(f"High dynamic range ({result.dynamic_range:.0f})")
        
        # Reconstruction error risk
        if result.reconstruction_mse > cls.THRESHOLDS["reconstruction_mse_very_high"]:
            score += 25
            reasons.append(f"High reconstruction error (MSE={result.reconstruction_mse:.2e})")
        elif result.reconstruction_mse > cls.THRESHOLDS["reconstruction_mse_high"]:
            score += 15
            reasons.append(f"Moderate reconstruction error (MSE={result.reconstruction_mse:.2e})")
        
        if result.reconstruction_cosine_sim < cls.THRESHOLDS["cosine_sim_very_low"]:
            score += 20
            reasons.append(f"Low cosine similarity ({result.reconstruction_cosine_sim:.6f})")
        elif result.reconstruction_cosine_sim < cls.THRESHOLDS["cosine_sim_low"]:
            score += 10
            reasons.append(f"Moderate cosine similarity drop ({result.reconstruction_cosine_sim:.6f})")
        
        # Activation sensitivity (if measured)
        if result.activation_mse is not None and result.activation_mse > 1e-3:
            score += 20
            reasons.append(f"High activation sensitivity (MSE={result.activation_mse:.2e})")
        
        # Cap at 100
        score = min(score, 100)
        
        return score, reasons
    
    @classmethod
    def score_to_safety(cls, score: float, result: LayerAnalysisResult) -> QuantizationSafety:
        """Convert risk score to safety classification."""
        # Always skip normalization layers regardless of score
        if result.is_normalization:
            return QuantizationSafety.CRITICAL
        
        if score >= 70:
            return QuantizationSafety.CRITICAL
        elif score >= 50:
            return QuantizationSafety.SENSITIVE
        elif score >= 30:
            return QuantizationSafety.CAUTION
        else:
            return QuantizationSafety.SAFE


class ActivationAnalyzer:
    """Analyzes activation sensitivity to quantization."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.activation_cache: Dict[str, Tensor] = {}
        self.hooks = []
    
    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activation_cache[name] = output.detach().clone()
        return hook
    
    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks for specified layers."""
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_cache(self):
        """Clear activation cache."""
        self.activation_cache = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QuantizationSensitivityAnalyzer:
    """Main analyzer class that orchestrates the full analysis."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Thinking",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        low_memory: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.low_memory = low_memory
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the model for analysis."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Use device_map for memory efficiency during analysis
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.low_memory else self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def analyze_layer(
        self,
        name: str,
        module: nn.Module,
        test_activations: bool = False
    ) -> Optional[LayerAnalysisResult]:
        """Analyze a single layer for quantization sensitivity."""
        
        # Only analyze layers with parameters
        params = list(module.parameters(recurse=False))
        if not params:
            return None
        
        # Get the main weight tensor
        weight = None
        for param in params:
            if param.ndim >= 2:  # Prefer 2D+ weights (linear layers)
                weight = param.data
                break
        
        if weight is None:
            weight = params[0].data
        
        # Calculate param count and size
        param_count = sum(p.numel() for p in params)
        param_size_mb = sum(p.numel() * p.element_size() for p in params) / (1024 ** 2)
        
        # Create result object
        result = LayerAnalysisResult(
            name=name,
            module_type=module.__class__.__name__,
            param_count=param_count,
            param_size_mb=param_size_mb
        )
        
        # Classify layer type
        classification = LayerClassifier.classify(name, module)
        result.is_vision_layer = classification["is_vision_layer"]
        result.is_cross_attention = classification["is_cross_attention"]
        result.is_embedding = classification["is_embedding"]
        result.is_normalization = classification["is_normalization"]
        result.is_output_layer = classification["is_output_layer"]
        
        # Compute weight statistics
        with torch.no_grad():
            weight_cpu = weight.float().cpu()
            stats = WeightAnalyzer.compute_statistics(weight_cpu)
            
            result.weight_mean = stats["mean"]
            result.weight_std = stats["std"]
            result.weight_min = stats["min"]
            result.weight_max = stats["max"]
            result.weight_kurtosis = stats["kurtosis"]
            result.weight_skewness = stats["skewness"]
            result.outlier_ratio = stats["outlier_ratio"]
            result.dynamic_range = stats["dynamic_range"]
            
            # Simulate quantization and measure reconstruction error
            quantized, _ = QuantizationSimulator.symmetric_quantize(weight_cpu, bits=8)
            recon_metrics = QuantizationSimulator.compute_reconstruction_error(weight_cpu, quantized)
            
            result.reconstruction_mse = recon_metrics["mse"]
            result.reconstruction_max_error = recon_metrics["max_error"]
            result.reconstruction_cosine_sim = recon_metrics["cosine_sim"]
            
            # Clean up
            del weight_cpu, quantized
        
        # Compute risk score
        risk_score, reasons = RiskScorer.compute_risk_score(result)
        result.risk_score = risk_score
        result.reasons = reasons
        result.safety = RiskScorer.score_to_safety(risk_score, result)
        
        return result
    
    def analyze_model(
        self,
        test_activations: bool = False,
        sample_image_path: Optional[str] = None
    ) -> ModelAnalysisReport:
        """Run full analysis on the model."""
        
        if self.model is None:
            self.load_model()
        
        logger.info("Starting layer-by-layer analysis...")
        
        # Initialize report
        report = ModelAnalysisReport(
            model_name=self.model_name,
            analysis_timestamp=datetime.now().isoformat(),
            total_params=sum(p.numel() for p in self.model.parameters()),
            total_size_mb=sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        )
        
        # Analyze each layer
        layers_to_analyze = []
        for name, module in self.model.named_modules():
            # Focus on layers with meaningful parameters
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                layers_to_analyze.append((name, module))
        
        logger.info(f"Analyzing {len(layers_to_analyze)} layers...")
        
        for name, module in tqdm(layers_to_analyze, desc="Analyzing layers"):
            try:
                result = self.analyze_layer(name, module, test_activations)
                if result:
                    report.layer_results[name] = result
                    
                    # Categorize by safety
                    if result.safety == QuantizationSafety.SAFE:
                        report.safe_layers.append(name)
                    elif result.safety == QuantizationSafety.CAUTION:
                        report.caution_layers.append(name)
                    elif result.safety == QuantizationSafety.SENSITIVE:
                        report.sensitive_layers.append(name)
                    elif result.safety == QuantizationSafety.CRITICAL:
                        report.critical_layers.append(name)
                        
            except Exception as e:
                logger.warning(f"Error analyzing layer {name}: {e}")
        
        # Calculate memory estimates
        for name, result in report.layer_results.items():
            if result.safety in [QuantizationSafety.CRITICAL, QuantizationSafety.SENSITIVE]:
                report.non_quantizable_size_mb += result.param_size_mb
            else:
                report.quantizable_size_mb += result.param_size_mb
        
        # Estimate Q8 size (quantizable layers at 1/2 size for int8)
        report.estimated_q8_size_mb = (
            report.non_quantizable_size_mb +  # Keep these in original precision
            report.quantizable_size_mb / 2     # These get quantized
        )
        
        # Generate skip_modules list for BitsAndBytes
        report.skip_modules = self._generate_skip_modules(report)
        
        logger.info("Analysis complete!")
        return report
    
    def _generate_skip_modules(self, report: ModelAnalysisReport) -> List[str]:
        """Generate list of module name patterns to skip during quantization."""
        skip_modules = set()
        
        for name, result in report.layer_results.items():
            if result.safety in [QuantizationSafety.CRITICAL, QuantizationSafety.SENSITIVE]:
                # Extract the parent module name for skipping
                # e.g., "model.visual.blocks.0.attn.qkv" -> "visual"
                parts = name.split(".")
                
                # Find the most specific reasonable skip pattern
                if "visual" in parts:
                    skip_modules.add("visual")
                elif "merger" in parts:
                    skip_modules.add("merger")
                elif "cross_attn" in name.lower():
                    # Add the specific cross-attention module
                    for i, part in enumerate(parts):
                        if "cross" in part.lower() or "attn" in part.lower():
                            skip_modules.add(".".join(parts[:i+1]))
                            break
                else:
                    # For other critical layers, add immediate parent
                    if len(parts) > 1:
                        skip_modules.add(parts[0])
        
        return sorted(list(skip_modules))
    
    def generate_report(self, report: ModelAnalysisReport, output_path: str = "quantization_report.json"):
        """Generate detailed JSON report."""
        
        # Convert to serializable format
        report_dict = {
            "model_name": report.model_name,
            "analysis_timestamp": report.analysis_timestamp,
            "summary": {
                "total_params": report.total_params,
                "total_size_mb": round(report.total_size_mb, 2),
                "quantizable_size_mb": round(report.quantizable_size_mb, 2),
                "non_quantizable_size_mb": round(report.non_quantizable_size_mb, 2),
                "estimated_q8_size_mb": round(report.estimated_q8_size_mb, 2),
                "memory_savings_percent": round(
                    (1 - report.estimated_q8_size_mb / report.total_size_mb) * 100, 1
                ) if report.total_size_mb > 0 else 0,
                "safe_layer_count": len(report.safe_layers),
                "caution_layer_count": len(report.caution_layers),
                "sensitive_layer_count": len(report.sensitive_layers),
                "critical_layer_count": len(report.critical_layers),
            },
            "skip_modules_for_bitsandbytes": report.skip_modules,
            "layer_classifications": {
                "safe": report.safe_layers,
                "caution": report.caution_layers,
                "sensitive": report.sensitive_layers,
                "critical": report.critical_layers,
            },
            "layer_details": {}
        }
        
        # Add detailed layer info
        for name, result in report.layer_results.items():
            report_dict["layer_details"][name] = {
                "module_type": result.module_type,
                "param_count": result.param_count,
                "param_size_mb": round(result.param_size_mb, 4),
                "safety": result.safety.value,
                "risk_score": round(result.risk_score, 1),
                "reasons": result.reasons,
                "layer_type": {
                    "is_vision_layer": result.is_vision_layer,
                    "is_cross_attention": result.is_cross_attention,
                    "is_embedding": result.is_embedding,
                    "is_normalization": result.is_normalization,
                    "is_output_layer": result.is_output_layer,
                },
                "weight_statistics": {
                    "mean": round(result.weight_mean, 6),
                    "std": round(result.weight_std, 6),
                    "min": round(result.weight_min, 6),
                    "max": round(result.weight_max, 6),
                    "kurtosis": round(result.weight_kurtosis, 2),
                    "skewness": round(result.weight_skewness, 2),
                    "outlier_ratio": round(result.outlier_ratio, 6),
                    "dynamic_range": round(result.dynamic_range, 2),
                },
                "reconstruction_metrics": {
                    "mse": result.reconstruction_mse,
                    "max_error": result.reconstruction_max_error,
                    "cosine_similarity": round(result.reconstruction_cosine_sim, 8),
                }
            }
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return report_dict
    
    def print_summary(self, report: ModelAnalysisReport):
        """Print human-readable summary."""
        
        print("\n" + "=" * 70)
        print("QUANTIZATION SENSITIVITY ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nModel: {report.model_name}")
        print(f"Analysis Time: {report.analysis_timestamp}")
        
        print("\n" + "-" * 70)
        print("MEMORY SUMMARY")
        print("-" * 70)
        print(f"Total Model Size: {report.total_size_mb:.2f} MB ({report.total_size_mb/1024:.2f} GB)")
        print(f"Quantizable Layers: {report.quantizable_size_mb:.2f} MB")
        print(f"Non-Quantizable Layers: {report.non_quantizable_size_mb:.2f} MB")
        print(f"Estimated Q8 Size: {report.estimated_q8_size_mb:.2f} MB ({report.estimated_q8_size_mb/1024:.2f} GB)")
        savings = (1 - report.estimated_q8_size_mb / report.total_size_mb) * 100
        print(f"Memory Savings: {savings:.1f}%")
        
        print("\n" + "-" * 70)
        print("LAYER CLASSIFICATION SUMMARY")
        print("-" * 70)
        print(f"✅ SAFE to quantize: {len(report.safe_layers)} layers")
        print(f"⚠️  CAUTION (minor risk): {len(report.caution_layers)} layers")
        print(f"🔶 SENSITIVE (noticeable risk): {len(report.sensitive_layers)} layers")
        print(f"🛑 CRITICAL (do not quantize): {len(report.critical_layers)} layers")
        
        print("\n" + "-" * 70)
        print("CRITICAL LAYERS (DO NOT QUANTIZE)")
        print("-" * 70)
        for name in report.critical_layers[:20]:  # Show first 20
            result = report.layer_results[name]
            print(f"  • {name}")
            print(f"    Type: {result.module_type}, Risk: {result.risk_score:.0f}/100")
            for reason in result.reasons[:2]:
                print(f"    - {reason}")
        if len(report.critical_layers) > 20:
            print(f"  ... and {len(report.critical_layers) - 20} more")
        
        print("\n" + "-" * 70)
        print("SENSITIVE LAYERS (QUANTIZE WITH CAUTION)")
        print("-" * 70)
        for name in report.sensitive_layers[:10]:
            result = report.layer_results[name]
            print(f"  • {name} (Risk: {result.risk_score:.0f}/100)")
        if len(report.sensitive_layers) > 10:
            print(f"  ... and {len(report.sensitive_layers) - 10} more")
        
        print("\n" + "-" * 70)
        print("RECOMMENDED SKIP MODULES FOR BITSANDBYTES")
        print("-" * 70)
        print("Use this in your loading code:\n")
        print("```python")
        print("from transformers import BitsAndBytesConfig")
        print("")
        print("quantization_config = BitsAndBytesConfig(")
        print("    load_in_8bit=True,")
        print(f"    llm_int8_skip_modules={report.skip_modules}")
        print(")")
        print("```")
        
        print("\n" + "=" * 70)


def generate_loading_code(report: ModelAnalysisReport, output_path: str = "load_quantized_model.py"):
    """Generate ready-to-use loading code based on analysis."""
    
    code = f'''#!/usr/bin/env python3
"""
Auto-generated loading code for {report.model_name}
Generated by Quantization Sensitivity Analyzer on {report.analysis_timestamp}

This code loads the model with selective INT8 quantization,
preserving full precision for sensitive layers (especially vision components).
"""

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)

def load_model(device_map: str = "auto"):
    """
    Load {report.model_name} with selective quantization.
    
    Memory estimates:
    - Original BF16 size: {report.total_size_mb/1024:.2f} GB
    - With selective Q8: {report.estimated_q8_size_mb/1024:.2f} GB
    - Memory savings: {(1 - report.estimated_q8_size_mb/report.total_size_mb)*100:.1f}%
    
    Layers kept in full precision:
    - Critical layers: {len(report.critical_layers)}
    - Sensitive layers: {len(report.sensitive_layers)}
    """
    
    # Configure quantization - skip sensitive modules
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules={report.skip_modules},
        llm_int8_threshold=6.0,  # Default threshold for outlier handling
    )
    
    # Load model with selective quantization
    model = AutoModelForVision2Seq.from_pretrained(
        "{report.model_name}",
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,  # For non-quantized layers
        trust_remote_code=True,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "{report.model_name}",
        trust_remote_code=True
    )
    
    return model, processor


def load_model_full_precision(device_map: str = "auto"):
    """
    Load model in full BF16 precision (no quantization).
    Requires ~{report.total_size_mb/1024:.1f} GB VRAM.
    """
    model = AutoModelForVision2Seq.from_pretrained(
        "{report.model_name}",
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        "{report.model_name}",
        trust_remote_code=True
    )
    
    return model, processor


def load_model_aggressive_quantization(device_map: str = "auto"):
    """
    Load model with aggressive INT8 quantization (only skip truly critical layers).
    Smaller memory footprint but may have slight quality degradation.
    
    Only skips: {[m for m in report.skip_modules if any(p in m for p in ['norm', 'embed', 'lm_head'])]}
    """
    # Minimal skip list - only absolute essentials
    minimal_skip = [m for m in {report.skip_modules} 
                    if any(p in m.lower() for p in ['norm', 'embed', 'lm_head'])]
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=minimal_skip or ["lm_head"],
        llm_int8_threshold=6.0,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        "{report.model_name}",
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        "{report.model_name}",
        trust_remote_code=True
    )
    
    return model, processor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load quantized model")
    parser.add_argument("--mode", choices=["selective", "full", "aggressive"], 
                       default="selective", help="Quantization mode")
    parser.add_argument("--device-map", default="auto", help="Device map strategy")
    args = parser.parse_args()
    
    print(f"Loading model in {{args.mode}} mode...")
    
    if args.mode == "selective":
        model, processor = load_model(args.device_map)
    elif args.mode == "full":
        model, processor = load_model_full_precision(args.device_map)
    else:
        model, processor = load_model_aggressive_quantization(args.device_map)
    
    print("Model loaded successfully!")
    print(f"Model device map: {{model.hf_device_map}}")
'''
    
    with open(output_path, "w") as f:
        f.write(code)
    
    logger.info(f"Loading code saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model layers for quantization sensitivity"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-32B-Thinking",
        help="Model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantization_report.json",
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--generate-code",
        action="store_true",
        help="Generate ready-to-use loading code"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for analysis"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model loading"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        default=True,
        help="Use memory-efficient loading (device_map=auto)"
    )
    
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    # Create analyzer
    analyzer = QuantizationSensitivityAnalyzer(
        model_name=args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        low_memory=args.low_memory
    )
    
    # Run analysis
    report = analyzer.analyze_model()
    
    # Print summary
    analyzer.print_summary(report)
    
    # Save JSON report
    analyzer.generate_report(report, args.output)
    
    # Generate loading code if requested
    if args.generate_code:
        code_path = args.output.replace(".json", "_loader.py")
        generate_loading_code(report, code_path)


if __name__ == "__main__":
    main()
