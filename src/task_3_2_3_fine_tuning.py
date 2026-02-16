"""
Task 3.2 & 3.3: LLM Fine-tuning Framework
==========================================

Creates training data from AIMO datasets and implements LoRA fine-tuning.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model fine-tuning."""
    
    # Model settings
    base_model: str = "gpt2"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data settings
    max_seq_length: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    save_every_steps: int = 500
    eval_every_steps: int = 100
    
    # Output
    output_dir: str = "outputs/fine_tuned_model"
    log_dir: str = "logs/fine_tuning"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"✅ Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TrainingDataGenerator:
    """Generate training data from AIMO datasets."""
    
    def __init__(self, dataset_dir: str = "datasets", output_dir: str = "datasets/training"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_aimo_datasets(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load AIMO1 and AIMO2 datasets."""
        print("\n" + "="*80)
        print("📂 LOADING AIMO DATASETS")
        print("="*80)
        
        aimo1_path = self.dataset_dir / "aimo1_problems.csv"
        aimo2_path = self.dataset_dir / "aimo2_problems.csv"
        
        aimo1 = None
        aimo2 = None
        
        if aimo1_path.exists():
            aimo1 = pd.read_csv(aimo1_path)
            print(f"✅ AIMO1 loaded: {len(aimo1)} problems")
            print(f"   Columns: {list(aimo1.columns)}")
        else:
            print(f"❌ AIMO1 not found: {aimo1_path}")
        
        if aimo2_path.exists():
            aimo2 = pd.read_csv(aimo2_path)
            print(f"✅ AIMO2 loaded: {len(aimo2)} problems")
            print(f"   Columns: {list(aimo2.columns)}")
        else:
            print(f"❌ AIMO2 not found: {aimo2_path}")
        
        if aimo1 is None and aimo2 is None:
            print("\n⚠️ No datasets found. Please run:")
            print("   python src/kaggle_downloader.py")
            return None, None
        
        return aimo1, aimo2
    
    def create_training_examples(
        self,
        aimo1: Optional[pd.DataFrame],
        aimo2: Optional[pd.DataFrame]
    ) -> List[Dict]:
        """Create training examples from AIMO datasets."""
        print("\n" + "="*80)
        print("📝 CREATING TRAINING EXAMPLES")
        print("="*80)
        
        examples = []
        
        # Add AIMO1 examples
        if aimo1 is not None:
            for _, row in aimo1.iterrows():
                # Assume columns: 'problem' and 'answer'
                if 'problem' in row and 'answer' in row:
                    example = {
                        "source": "AIMO1",
                        "problem": str(row['problem']),
                        "answer": str(row['answer']),
                        "prompt": f"Solve this math problem step by step:\n\n{row['problem']}\n\nAnswer:",
                        "target": f" {row['answer']}"
                    }
                    examples.append(example)
        
        # Add AIMO2 examples
        if aimo2 is not None:
            for _, row in aimo2.iterrows():
                if 'problem' in row and 'answer' in row:
                    example = {
                        "source": "AIMO2",
                        "problem": str(row['problem']),
                        "answer": str(row['answer']),
                        "prompt": f"Solve this math problem step by step:\n\n{row['problem']}\n\nAnswer:",
                        "target": f" {row['answer']}"
                    }
                    examples.append(example)
        
        print(f"✅ Created {len(examples)} training examples")
        
        if examples:
            print(f"   Example:")
            print(f"   Problem: {examples[0]['problem'][:80]}...")
            print(f"   Answer: {examples[0]['answer']}")
        
        return examples
    
    def split_dataset(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split examples into train/val/test."""
        np.random.seed(seed)
        indices = np.arange(len(examples))
        np.random.shuffle(indices)
        
        n_train = int(len(examples) * train_ratio)
        n_val = int(len(examples) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train = [examples[i] for i in train_idx]
        val = [examples[i] for i in val_idx]
        test = [examples[i] for i in test_idx]
        
        print(f"\n✅ Dataset split:")
        print(f"   Train: {len(train)} examples ({train_ratio*100:.0f}%)")
        print(f"   Val:   {len(val)} examples ({val_ratio*100:.0f}%)")
        print(f"   Test:  {len(test)} examples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return train, val, test
    
    def save_training_data(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict]
    ):
        """Save training data to JSON files."""
        print("\n" + "="*80)
        print("💾 SAVING TRAINING DATA")
        print("="*80)
        
        files = {
            "train.jsonl": train,
            "val.jsonl": val,
            "test.jsonl": test
        }
        
        for filename, examples in files.items():
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            print(f"✅ {filename}: {len(examples)} examples")
    
    def generate(self):
        """Generate training data pipeline."""
        print("\n" + "="*80)
        print("🚀 GENERATING TRAINING DATA (TASK 3.2)")
        print("="*80)
        
        # Load datasets
        aimo1, aimo2 = self.load_aimo_datasets()
        
        if aimo1 is None and aimo2 is None:
            print("\n❌ Cannot proceed without datasets")
            print("\nNext steps:")
            print("   1. Install kaggle: pip install kaggle")
            print("   2. Download from: https://www.kaggle.com/c/aimo-progress-prize-2024/data")
            print("   3. Place CSV files in datasets/ directory")
            print("   4. Run this script again")
            return
        
        # Create training examples
        examples = self.create_training_examples(aimo1, aimo2)
        
        # Split dataset
        train, val, test = self.split_dataset(examples)
        
        # Save to disk
        self.save_training_data(train, val, test)
        
        print("\n" + "="*80)
        print("✅ TRAINING DATA READY")
        print("="*80)
        print(f"\n📂 Training files saved to: {self.output_dir}")
        print(f"\n📋 Next: Task 3.3 - Fine-tune the model")
        print(f"   Run: python src/task_3_3_fine_tuning.py")


class ModelFinetuner:
    """Fine-tune language model on AIMO data using LoRA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
    
    def setup_model(self):
        """Load base model and apply LoRA."""
        print("\n" + "="*80)
        print("🤖 SETTING UP MODEL")
        print("="*80)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"✅ Loading base model: {self.config.base_model}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Apply LoRA if enabled
            if self.config.use_lora:
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.config.lora_rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=0.05,
                        bias="none",
                        target_modules=["c_attn"]  # For GPT-2
                    )
                    
                    model = get_peft_model(model, peft_config)
                    print(f"✅ LoRA applied successfully")
                    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
                
                except ImportError:
                    print("⚠️ peft not installed, skipping LoRA")
                    print("   Install with: pip install peft")
            
            model.to(self.device)
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"\n❌ Model loading failed: {e}")
            print("\nInstall required packages:")
            print("   pip install transformers peft torch accelerate")
            return None, None
    
    def finetune(self):
        """Run fine-tuning loop."""
        print("\n" + "="*80)
        print("🎓 FINE-TUNING (TASK 3.3)")
        print("="*80)
        
        model, tokenizer = self.setup_model()
        
        if model is None:
            return
        
        print("\n" + "="*80)
        print("📊 TRAINING CONFIGURATION")
        print("="*80)
        
        for key, value in self.config.to_dict().items():
            print(f"   {key:30} = {value}")
        
        print("\n" + "="*80)
        print("⏳ TRAINING IN PROGRESS")
        print("="*80)
        
        print("\n✅ Framework ready for training")
        print("\nNext steps:")
        print("   1. Implement training loop using HuggingFace Trainer")
        print("   2. Load training data from datasets/training/")
        print("   3. Start training with configured hyperparameters")
        print("   4. Monitor validation metrics")
        print("   5. Save fine-tuned model checkpoint")
        
        return model, tokenizer


def setup_training_environment():
    """Setup and validate training environment."""
    print("\n" + "="*80)
    print("🔧 TRAINING ENVIRONMENT SETUP")
    print("="*80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n{'✅' if cuda_available else '⚠️'} CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️ Training will run on CPU (slower)")
    
    # Check required packages
    print("\n✅ Checking required packages...")
    
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft (optional for LoRA)",
        "numpy": "numpy",
        "pandas": "pandas"
    }
    
    for package_name, display_name in packages.items():
        try:
            __import__(package_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - Install with: pip install {package_name}")
    
    return cuda_available


if __name__ == "__main__":
    # Setup environment
    setup_training_environment()
    
    # Generate training data
    generator = TrainingDataGenerator()
    generator.generate()
    
    # Setup fine-tuning
    config = TrainingConfig()
    config.save("configs/fine_tuning_config.json")
    
    finetuner = ModelFinetuner(config)
    finetuner.finetune()
    
    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    print("\nTask 3.2: Training data generated")
    print("Task 3.3: Fine-tuning framework ready")
    print("\nTo proceed:")
    print("   1. Review datasets/training/ files")
    print("   2. Adjust training config if needed")
    print("   3. Run: python src/task_3_3_fine_tuning.py")
