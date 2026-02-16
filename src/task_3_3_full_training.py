"""
Task 3.3: Full LLM Fine-tuning Implementation
=============================================

Complete fine-tuning pipeline using HuggingFace Trainer with LoRA.
Trains GPT-2 on AIMO problems from prepared JSONL datasets.
"""

import json
import torch
import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    """Training arguments configuration."""
    output_dir: str = "outputs/fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True  # Mixed precision
    seed: int = 42
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    push_to_hub: bool = False


class AIOMTrainer:
    """Full training pipeline for AIMO problem solving model."""
    
    def __init__(self, config_path: str = "configs/fine_tuning_config.json"):
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path("datasets/training")
        self.output_dir = Path("outputs/fine_tuned_model")
        
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    def load_jsonl_dataset(self, filepath: Path):
        """Load JSONL training data."""
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples
    
    def prepare_datasets(self):
        """Load and prepare training datasets."""
        print("\n" + "="*80)
        print("📂 LOADING DATASETS")
        print("="*80)
        
        # Load datasets
        train_data = self.load_jsonl_dataset(self.data_dir / "train.jsonl")
        val_data = self.load_jsonl_dataset(self.data_dir / "val.jsonl")
        test_data = self.load_jsonl_dataset(self.data_dir / "test.jsonl")
        
        print(f"\n✅ Train: {len(train_data)} examples")
        print(f"✅ Val: {len(val_data)} examples")
        print(f"✅ Test: {len(test_data)} examples")
        
        if train_data:
            print(f"\n📝 Sample training example:")
            sample = train_data[0]
            print(f"   Problem: {sample['problem'][:80]}...")
            print(f"   Answer: {sample['answer']}")
        
        return train_data, val_data, test_data
    
    def setup_model_and_tokenizer(self):
        """Load model and tokenizer."""
        print("\n" + "="*80)
        print("🤖 LOADING MODEL & TOKENIZER")
        print("="*80)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import get_peft_model, LoraConfig, TaskType
            
            # Load tokenizer
            model_name = "gpt2"
            print(f"\n📥 Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
            
            # Load model
            print(f"📥 Loading model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
            
            # Apply LoRA
            print(f"\n🔧 Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_attn"],  # For GPT-2
            )
            model = get_peft_model(model, lora_config)
            
            # Count trainable params
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"✅ LoRA applied")
            print(f"   Total parameters: {total:,}")
            print(f"   Trainable parameters: {trainable:,}")
            print(f"   Efficiency: {100 * (1 - trainable/total):.2f}% reduction")
            
            model.to(self.device)
            return model, tokenizer
        
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            print("\n❌ Installation required:")
            print("   pip install transformers peft torch")
            raise
    
    def tokenize_function(self, examples, tokenizer, max_length=512):
        """Tokenize training examples."""
        # Create full text: prompt + target
        texts = [ex["prompt"] + ex["target"] for ex in examples]
        
        # Tokenize
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare labels (same as input for CLM)
        encodings["labels"] = encodings["input_ids"].clone()
        
        return encodings
    
    def create_trainer(self, model, tokenizer, train_data, val_data):
        """Create HuggingFace Trainer."""
        print("\n" + "="*80)
        print("⚙️ SETTING UP TRAINER")
        print("="*80)
        
        try:
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            from datasets import Dataset
            
            # Create HF datasets
            train_dataset = Dataset.from_dict({
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            })
            val_dataset = Dataset.from_dict({
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            })
            
            # Tokenize datasets
            print("\n📝 Tokenizing training data...")
            train_encodings = self.tokenize_function(train_data, tokenizer)
            val_encodings = self.tokenize_function(val_data, tokenizer)
            
            # Create proper datasets
            train_dataset = Dataset.from_dict(train_encodings)
            val_dataset = Dataset.from_dict(val_encodings)
            
            print(f"✅ Train dataset: {len(train_dataset)} examples")
            print(f"✅ Val dataset: {len(val_dataset)} examples")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=100,
                logging_steps=100,
                eval_strategy="steps",  # Match evaluation with save strategy
                eval_steps=500,
                save_strategy="steps",
                save_steps=500,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=torch.cuda.is_available(),
                seed=42,
                remove_unused_columns=False,
            )
            
            print(f"\n✅ Training arguments configured")
            print(f"   Output dir: {training_args.output_dir}")
            print(f"   Epochs: {training_args.num_train_epochs}")
            print(f"   Batch size: {training_args.per_device_train_batch_size}")
            print(f"   Learning rate: {training_args.learning_rate}")
            print(f"   Mixed precision: {training_args.fp16}")
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[]
            )
            
            return trainer, training_args
        
        except ImportError as e:
            logger.error(f"Failed to setup trainer: {e}")
            print("\n❌ Additional packages required:")
            print("   pip install datasets accelerate")
            raise
    
    def train(self):
        """Run full training pipeline."""
        print("\n" + "="*80)
        print("🎓 STARTING FINE-TUNING (TASK 3.3)")
        print("="*80)
        
        try:
            # Prepare data
            train_data, val_data, test_data = self.prepare_datasets()
            
            # Setup model
            model, tokenizer = self.setup_model_and_tokenizer()
            
            # Create trainer
            trainer, training_args = self.create_trainer(model, tokenizer, train_data, val_data)
            
            # Train
            print("\n" + "="*80)
            print("⏳ TRAINING IN PROGRESS")
            print("="*80)
            
            print("\n🚀 Starting training loop...")
            print(f"   Training on {len(train_data)} examples")
            print(f"   Validating on {len(val_data)} examples")
            print(f"   Estimated time: 2-4 hours on GPU, 12-24 hours on CPU")
            print(f"   Checkpoint interval: 500 steps")
            print(f"   Validation interval: 500 steps")
            
            train_result = trainer.train()
            
            # Save model
            print("\n" + "="*80)
            print("💾 SAVING MODEL")
            print("="*80)
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.output_dir))
            tokenizer.save_pretrained(str(self.output_dir))
            
            print(f"\n✅ Model saved to: {self.output_dir}")
            print(f"   Files: adapter_config.json, adapter_model.bin, config.json, tokenizer.json")
            
            # Evaluate on test set
            print("\n" + "="*80)
            print("📊 EVALUATING ON TEST SET")
            print("="*80)
            
            test_dataset = trainer.dataset.from_dict({
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            })
            test_encodings = self.tokenize_function(test_data, tokenizer)
            test_dataset = trainer.dataset.from_dict(test_encodings)
            
            test_results = trainer.evaluate(eval_dataset=test_dataset)
            print(f"\n✅ Test Results:")
            print(f"   Test Loss: {test_results.get('eval_loss', 'N/A'):.4f}")
            
            # Summary
            print("\n" + "="*80)
            print("✨ TRAINING COMPLETE")
            print("="*80)
            
            print(f"\n📈 Training Results:")
            print(f"   Final Loss: {train_result.training_loss:.4f}")
            print(f"   Training completed at: {datetime.now()}")
            print(f"   Model saved to: {self.output_dir}")
            
            print(f"\n📋 Next Steps:")
            print(f"   1. Test inference on sample problems")
            print(f"   2. Task 3.4: Optimize prompting")
            print(f"   3. Task 3.5: Implement ensemble methods")
            print(f"   4. Task 3.6: Validate and submit")
            
            return trainer, model, tokenizer
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"\n❌ Training error: {e}")
            raise


def main():
    """Main training script."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "🎓 PHASE 3.3: FULL FINE-TUNING" + " "*28 + "║")
    print("║" + " "*15 + "Complete LLM Fine-tuning on AIMO Problems" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    # Check if datasets exist
    data_dir = Path("datasets/training")
    if not (data_dir / "train.jsonl").exists():
        print("\n❌ Training data not found!")
        print("   Please run: python src/task_3_2_3_fine_tuning.py")
        return
    
    # Run training
    trainer_obj = AIOMTrainer()
    try:
        trainer, model, tokenizer = trainer_obj.train()
        print("\n" + "="*80)
        print("✅ FINE-TUNING SUCCESSFUL!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Ensure all dependencies installed: pip install transformers peft torch datasets accelerate")
        print("   2. Check GPU memory: nvidia-smi")
        print("   3. Reduce batch size if OOM: modify training_args.per_device_train_batch_size")


if __name__ == "__main__":
    main()
