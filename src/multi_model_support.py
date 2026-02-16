"""
Multi-Model Support for AIMO3
Supports: DeepSeek-R1, Llama 4, Qwen 3, Gemma 3, GPT-2, Mistral Large 3, Kimi K2.5
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for each supported model"""
    name: str
    repo_id: str
    description: str
    max_tokens: int
    param_size: str
    reasoning_capable: bool
    vram_needed_gb: float
    inference_time_per_problem_sec: float
    supports_lora: bool
    notes: str


# Supported Models Registry
SUPPORTED_MODELS = {
    # Open-source reasoning models
    "deepseek-r1": ModelConfig(
        name="DeepSeek-R1",
        repo_id="deepseek-ai/DeepSeek-R1",
        description="Advanced reasoning model with strong mathematical capabilities",
        max_tokens=8192,
        param_size="67B",
        reasoning_capable=True,
        vram_needed_gb=180,
        inference_time_per_problem_sec=45,
        supports_lora=True,
        notes="Excellent for math reasoning, requires quantization for consumer GPUs"
    ),
    
    "llama4-scout": ModelConfig(
        name="Llama 4 Scout",
        repo_id="meta-llama/Llama-4-Scout",
        description="Efficient 8B variant optimized for reasoning",
        max_tokens=4096,
        param_size="8B",
        reasoning_capable=True,
        vram_needed_gb=16,
        inference_time_per_problem_sec=8,
        supports_lora=True,
        notes="Good balance of capability and efficiency"
    ),
    
    "llama4-maverick": ModelConfig(
        name="Llama 4 Maverick",
        repo_id="meta-llama/Llama-4-Maverick",
        description="Larger 70B variant for maximum capability",
        max_tokens=8192,
        param_size="70B",
        reasoning_capable=True,
        vram_needed_gb=160,
        inference_time_per_problem_sec=35,
        supports_lora=True,
        notes="Strongest Llama variant, requires quantization"
    ),
    
    # Qwen 3 variants
    "qwen3-32b": ModelConfig(
        name="Qwen 3 32B",
        repo_id="Qwen/Qwen3-32B",
        description="Balanced reasoning model",
        max_tokens=4096,
        param_size="32B",
        reasoning_capable=True,
        vram_needed_gb=64,
        inference_time_per_problem_sec=15,
        supports_lora=True,
        notes="Good performance on math problems"
    ),
    
    "qwen3-72b": ModelConfig(
        name="Qwen 3 72B",
        repo_id="Qwen/Qwen3-72B",
        description="Large-scale reasoning capability",
        max_tokens=8192,
        param_size="72B",
        reasoning_capable=True,
        vram_needed_gb=160,
        inference_time_per_problem_sec=40,
        supports_lora=True,
        notes="Excellent reasoning, requires quantization"
    ),
    
    "qwen3-235b": ModelConfig(
        name="Qwen 3 235B",
        repo_id="Qwen/Qwen3-235B",
        description="Largest Qwen model with strongest capabilities",
        max_tokens=16384,
        param_size="235B",
        reasoning_capable=True,
        vram_needed_gb=512,
        inference_time_per_problem_sec=120,
        supports_lora=False,
        notes="Premium model, available via API only"
    ),
    
    # Gemma 3 variants
    "gemma3-4b": ModelConfig(
        name="Gemma 3 4B",
        repo_id="google/gemma-3-4b",
        description="Lightweight efficient model",
        max_tokens=2048,
        param_size="4B",
        reasoning_capable=False,
        vram_needed_gb=8,
        inference_time_per_problem_sec=3,
        supports_lora=True,
        notes="Fast inference, limited reasoning"
    ),
    
    "gemma3-12b": ModelConfig(
        name="Gemma 3 12B",
        repo_id="google/gemma-3-12b",
        description="Balanced Gemma variant",
        max_tokens=4096,
        param_size="12B",
        reasoning_capable=True,
        vram_needed_gb=24,
        inference_time_per_problem_sec=6,
        supports_lora=True,
        notes="Good reasoning at reasonable cost"
    ),
    
    "gemma3-27b": ModelConfig(
        name="Gemma 3 27B",
        repo_id="google/gemma-3-27b",
        description="Larger Gemma with strong performance",
        max_tokens=8192,
        param_size="27B",
        reasoning_capable=True,
        vram_needed_gb=56,
        inference_time_per_problem_sec=12,
        supports_lora=True,
        notes="Good balance of performance and cost"
    ),
    
    # Legacy/baseline
    "gpt2": ModelConfig(
        name="GPT-2",
        repo_id="gpt2",
        description="Baseline model for testing",
        max_tokens=1024,
        param_size="124M",
        reasoning_capable=False,
        vram_needed_gb=1,
        inference_time_per_problem_sec=1,
        supports_lora=True,
        notes="Use for testing pipeline only"
    ),
    
    # Large open-source models
    "gpt-oss-120b": ModelConfig(
        name="GPT-OSS-120B",
        repo_id="gpt-oss/GPT-OSS-120B",
        description="Open-source 120B model with strong generalization",
        max_tokens=8192,
        param_size="120B",
        reasoning_capable=True,
        vram_needed_gb=280,
        inference_time_per_problem_sec=60,
        supports_lora=False,
        notes="Very large, requires enterprise hardware"
    ),
    
    # Mistral
    "mistral-large-3": ModelConfig(
        name="Mistral Large 3",
        repo_id="mistralai/Mistral-Large-3",
        description="Latest Mistral model with extended context",
        max_tokens=16384,
        param_size="123B",
        reasoning_capable=True,
        vram_needed_gb=280,
        inference_time_per_problem_sec=50,
        supports_lora=True,
        notes="Excellent reasoning, available via API"
    ),
    
    # Kimi (Chinese LLM with strong reasoning)
    "kimi-k2.5": ModelConfig(
        name="Kimi K2.5",
        repo_id="moonshot/kimi-k2.5",
        description="Chinese LLM with extended context and reasoning",
        max_tokens=200000,
        param_size="Unknown",
        reasoning_capable=True,
        vram_needed_gb=0,  # API-based
        inference_time_per_problem_sec=20,
        supports_lora=False,
        notes="API-based, strong reasoning capabilities, requires key"
    ),
}


class MultiModelLoader:
    """Load and manage multiple models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}
        self.tokenizers = {}
        
    def list_available_models(self) -> Dict[str, ModelConfig]:
        """List all available models"""
        return SUPPORTED_MODELS
    
    def get_model_info(self, model_key: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_key} not found. Available: {list(SUPPORTED_MODELS.keys())}")
        return SUPPORTED_MODELS[model_key]
    
    def load_model(self, model_key: str, lora_path: Optional[str] = None, 
                   quantize: bool = False) -> tuple:
        """
        Load a model and tokenizer
        
        Args:
            model_key: Key from SUPPORTED_MODELS
            lora_path: Optional path to LoRA weights
            quantize: Use 8-bit quantization (for large models)
            
        Returns:
            (model, tokenizer) tuple
        """
        if model_key in self.models:
            logger.info(f"Using cached model: {model_key}")
            return self.models[model_key], self.tokenizers[model_key]
        
        config = self.get_model_info(model_key)
        logger.info(f"Loading {config.name} ({config.param_size})...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.repo_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantize and torch.cuda.is_available():
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(config.repo_id, **kwargs)
        
        # Load LoRA if available
        if lora_path:
            logger.info(f"Loading LoRA from {lora_path}...")
            model = PeftModel.from_pretrained(model, lora_path)
        
        if self.device == "cpu" and "device_map" not in kwargs:
            model = model.to(self.device)
        
        model.eval()
        
        # Cache
        self.models[model_key] = model
        self.tokenizers[model_key] = tokenizer
        
        logger.info(f"✅ {config.name} loaded successfully")
        return model, tokenizer
    
    def unload_model(self, model_key: str) -> None:
        """Unload a model to free memory"""
        if model_key in self.models:
            del self.models[model_key]
            del self.tokenizers[model_key]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded {model_key}")


class APIBasedModel:
    """Interface for API-based models (Mistral, Kimi, etc)"""
    
    def __init__(self, model_key: str, api_key: str):
        self.model_key = model_key
        self.config = SUPPORTED_MODELS[model_key]
        self.api_key = api_key
        
    def generate(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response via API"""
        # To be implemented for specific APIs
        raise NotImplementedError(f"API support for {self.model_key} not yet implemented")


def compare_models_performance() -> None:
    """Print comparison table of all models"""
    print("\n" + "="*120)
    print("AIMO3 SUPPORTED MODELS COMPARISON")
    print("="*120)
    
    print(f"{'Model':<25} {'Size':<8} {'Reasoning':<10} {'VRAM':<10} {'Time/Q':<12} {'Notes':<30}")
    print("-"*120)
    
    for key, config in SUPPORTED_MODELS.items():
        reasoning = "✓" if config.reasoning_capable else "✗"
        print(f"{config.name:<25} {config.param_size:<8} {reasoning:<10} "
              f"{config.vram_needed_gb:<10.1f}GB {config.inference_time_per_problem_sec:<12.1f}s "
              f"{config.notes[:30]:<30}")
    
    print("="*120 + "\n")


def get_recommended_model(vram_gb: float, speed_priority: bool = False) -> str:
    """Get recommended model based on available VRAM"""
    available = [
        (key, cfg) for key, cfg in SUPPORTED_MODELS.items()
        if cfg.vram_needed_gb <= vram_gb and cfg.reasoning_capable
    ]
    
    if not available:
        logger.warning(f"No reasoning models fit in {vram_gb}GB VRAM, using GPT-2")
        return "gpt2"
    
    if speed_priority:
        # Sort by speed (fastest first)
        available.sort(key=lambda x: x[1].inference_time_per_problem_sec)
    else:
        # Sort by capability (largest first)
        param_order = {"4B": 4, "8B": 8, "12B": 12, "27B": 27, "32B": 32, "67B": 67, "70B": 70, "120B": 120}
        available.sort(key=lambda x: param_order.get(x[1].param_size, 0), reverse=True)
    
    chosen = available[0]
    logger.info(f"Recommended: {chosen[1].name} ({chosen[1].param_size})")
    return chosen[0]


if __name__ == "__main__":
    # Show all models
    compare_models_performance()
    
    # Test recommendation
    print("Recommendation for 24GB VRAM (speed priority):", get_recommended_model(24, speed_priority=True))
    print("Recommendation for 32GB VRAM (quality priority):", get_recommended_model(32, speed_priority=False))
    print("Recommendation for 64GB VRAM:", get_recommended_model(64, speed_priority=False))
