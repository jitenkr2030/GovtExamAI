#!/usr/bin/env python3
"""
Simplified Comprehensive Training Pipeline
Trains custom transformer model on 15 government exams with 685+ questions (Simplified Version)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedTrainingPipeline:
    def __init__(self, base_dir="/workspace"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "simplified_training_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load enhanced dataset
        dataset_path = self.base_dir / "data_collection/enhanced_exam_data/enhanced_exam_dataset.json"
        logger.info(f"Loading enhanced dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.enhanced_dataset = json.load(f)
        
        logger.info(f"Loaded {self.enhanced_dataset['total_questions']} questions from {self.enhanced_dataset['total_exam_types']} exam types")
        
        # Load original dataset for comparison
        original_dataset_path = self.base_dir / "training_outputs/processed_training_data.json"
        if original_dataset_path.exists():
            with open(original_dataset_path, 'r', encoding='utf-8') as f:
                self.original_dataset = json.load(f)
            logger.info("Loaded original dataset for comparison")
        else:
            self.original_dataset = None

    def prepare_training_data(self) -> Dict[str, Any]:
        """Prepare training data from all sources"""
        
        logger.info("🔄 Preparing training data...")
        
        # Combine enhanced dataset with original dataset
        all_questions = []
        
        # Add enhanced synthetic questions
        enhanced_questions = self.enhanced_dataset["questions"]
        for q in enhanced_questions:
            # Convert to training format
            training_question = {
                "input_text": f"Question: {q['question']}",
                "target": {
                    "subject": q["subject"],
                    "topic": q["topic"], 
                    "difficulty": q["difficulty"],
                    "correct_answer": q["correct_answer"]
                },
                "metadata": {
                    "exam_type": q["exam_type"],
                    "source": q["source"],
                    "question_id": q["question_id"],
                    "data_type": "enhanced_synthetic"
                }
            }
            all_questions.append(training_question)
        
        # Add original questions if available
        if self.original_dataset:
            for split in ["train", "validation", "test"]:
                if split in self.original_dataset:
                    for q in self.original_dataset[split]:
                        # Update to include enhanced metadata
                        q["metadata"]["data_type"] = "original_real"
                        all_questions.append(q)
        
        logger.info(f"Combined dataset: {len(all_questions)} total questions")
        
        # Create simple random splits (80/10/10)
        logger.info("📊 Creating train/validation/test splits...")
        
        # First split: train (80%) and temp (20%)
        train_questions, temp_questions = train_test_split(
            all_questions, 
            test_size=0.2, 
            random_state=42
        )
        
        # Second split: validation (10%) and test (10%) from temp
        val_questions, test_questions = train_test_split(
            temp_questions,
            test_size=0.5,
            random_state=42
        )
        
        # Create training data structure
        training_data = {
            "preparation_timestamp": datetime.now().isoformat(),
            "total_questions": len(all_questions),
            "original_questions": len(self.original_dataset["train"]) if self.original_dataset else 0,
            "enhanced_questions": len(enhanced_questions),
            "train": train_questions,
            "validation": val_questions,
            "test": test_questions,
            "stats": {
                "train_count": len(train_questions),
                "validation_count": len(val_questions),
                "test_count": len(test_questions),
                "exam_types": list(set([q["metadata"]["exam_type"] for q in all_questions])),
                "subjects": list(set([q["target"]["subject"] for q in all_questions])),
                "topics": list(set([q["target"]["topic"] for q in all_questions])),
                "difficulties": list(set([q["target"]["difficulty"] for q in all_questions]))
            }
        }
        
        # Save training data
        output_path = self.output_dir / "training_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Training data saved to: {output_path}")
        logger.info(f"  📚 Training: {len(train_questions)} questions")
        logger.info(f"  🔍 Validation: {len(val_questions)} questions")
        logger.info(f"  🧪 Test: {len(test_questions)} questions")
        
        return training_data

    def train_simplified_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train simplified model using basic transformers training"""
        
        logger.info("🚀 Starting simplified model training...")
        
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer, DataCollatorWithPadding,
                DistilBertConfig, DistilBertPreTrainedModel
            )
            from torch.utils.data import Dataset
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Model configuration
            model_name = "distilbert-base-uncased"
            logger.info(f"Using model: {model_name}")
            
            # Create label encoders
            all_questions = training_data["train"] + training_data["validation"] + training_data["test"]
            
            subjects = list(set([q["target"]["subject"] for q in all_questions]))
            topics = list(set([q["target"]["topic"] for q in all_questions]))
            difficulties = list(set([q["target"]["difficulty"] for q in all_questions]))
            
            subject_encoder = LabelEncoder().fit(subjects)
            topic_encoder = LabelEncoder().fit(topics)
            difficulty_encoder = LabelEncoder().fit(difficulties)
            
            logger.info(f"Label encoders created:")
            logger.info(f"  Subjects ({len(subjects)}): {subjects[:5]}...")
            logger.info(f"  Topics ({len(topics)}): {topics[:5]}...")
            logger.info(f"  Difficulties ({len(difficulties)}): {difficulties}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create a simple multi-task model
            class SimpleMultiTaskModel(DistilBertPreTrainedModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.distilbert = nn.Identity()  # We'll use a pre-trained model directly
                    
                    # Load pre-trained DistilBERT
                    from transformers import DistilBertModel
                    self.bert_model = DistilBertModel.from_pretrained(model_name)
                    
                    self.dropout = nn.Dropout(0.3)
                    
                    # Classification heads
                    self.subject_classifier = nn.Linear(config.hidden_size, len(subjects))
                    self.topic_classifier = nn.Linear(config.hidden_size, len(topics))
                    self.difficulty_classifier = nn.Linear(config.hidden_size, len(difficulties))
                    
                    self.init_weights()
                
                def forward(self, input_ids=None, attention_mask=None, subject_labels=None, topic_labels=None, difficulty_labels=None):
                    # Get BERT embeddings
                    outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
                    pooled_output = self.dropout(pooled_output)
                    
                    # Get predictions
                    subject_logits = self.subject_classifier(pooled_output)
                    topic_logits = self.topic_classifier(pooled_output)
                    difficulty_logits = self.difficulty_classifier(pooled_output)
                    
                    loss = None
                    # Check if we have individual labels
                    if hasattr(inputs, 'subject_labels'):
                        subject_loss = F.cross_entropy(subject_logits, inputs.subject_labels)
                        topic_loss = F.cross_entropy(topic_logits, inputs.topic_labels)
                        difficulty_loss = F.cross_entropy(difficulty_logits, inputs.difficulty_labels)
                        
                        # Combine losses
                        loss = subject_loss + topic_loss + difficulty_loss
                    elif "subject_labels" in inputs and "topic_labels" in inputs and "difficulty_labels" in inputs:
                        subject_loss = F.cross_entropy(subject_logits, inputs["subject_labels"])
                        topic_loss = F.cross_entropy(topic_logits, inputs["topic_labels"])
                        difficulty_loss = F.cross_entropy(difficulty_logits, inputs["difficulty_labels"])
                        
                        # Combine losses
                        loss = subject_loss + topic_loss + difficulty_loss
                    
                    return {
                        "loss": loss,
                        "subject_logits": subject_logits,
                        "topic_logits": topic_logits,
                        "difficulty_logits": difficulty_logits
                    }
            
            # Custom dataset class
            class SimpleDataset(Dataset):
                def __init__(self, questions, tokenizer, subject_encoder, topic_encoder, difficulty_encoder):
                    self.questions = questions
                    self.tokenizer = tokenizer
                    self.subject_encoder = subject_encoder
                    self.topic_encoder = topic_encoder
                    self.difficulty_encoder = difficulty_encoder
                
                def __len__(self):
                    return len(self.questions)
                
                def __getitem__(self, idx):
                    question = self.questions[idx]
                    text = question["input_text"]
                    
                    # Tokenize
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Encode labels
                    subject_label = self.subject_encoder.transform([question["target"]["subject"]])[0]
                    topic_label = self.topic_encoder.transform([question["target"]["topic"]])[0]
                    difficulty_label = self.difficulty_encoder.transform([question["target"]["difficulty"]])[0]
                    
                    return {
                        "input_ids": encoding["input_ids"].squeeze(),
                        "attention_mask": encoding["attention_mask"].squeeze(),
                        "subject_labels": torch.tensor(subject_label, dtype=torch.long),
                        "topic_labels": torch.tensor(topic_label, dtype=torch.long),
                        "difficulty_labels": torch.tensor(difficulty_label, dtype=torch.long)
                    }
            
            # Create datasets
            train_dataset = SimpleDataset(
                training_data["train"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder
            )
            val_dataset = SimpleDataset(
                training_data["validation"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder
            )
            test_dataset = SimpleDataset(
                training_data["test"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder
            )
            
            # Initialize model
            config = DistilBertConfig.from_pretrained(model_name)
            model = SimpleMultiTaskModel(config)
            
            logger.info("Model initialized successfully")
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "model_checkpoints"),
                num_train_epochs=5,  # Reduced epochs for faster training
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=200,
                weight_decay=0.01,
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=torch.cuda.is_available(),
                seed=42,
                remove_unused_columns=False
            )
            
            # Simple trainer with basic functionality
            class SimpleTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    # Extract individual labels
                    subject_labels = inputs.pop("subject_labels")
                    topic_labels = inputs.pop("topic_labels")
                    difficulty_labels = inputs.pop("difficulty_labels")
                    
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        subject_labels=subject_labels,
                        topic_labels=topic_labels,
                        difficulty_labels=difficulty_labels
                    )
                    loss = outputs["loss"]
                    return (loss, outputs) if return_outputs else loss
            
            # Initialize trainer
            trainer = SimpleTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )
            
            logger.info("Starting simplified model training...")
            
            # Train the model
            trainer.train()
            
            # Save the trained model
            model_save_path = self.output_dir / "simplified_exam_model"
            trainer.save_model(str(model_save_path))
            
            # Save encoders and configuration
            model_config = {
                "subjects": subjects,
                "topics": topics,
                "difficulties": difficulties,
                "subject_classes": subject_encoder.classes_.tolist(),
                "topic_classes": topic_encoder.classes_.tolist(),
                "difficulty_classes": difficulty_encoder.classes_.tolist(),
                "model_config": {
                    "model_name": model_name,
                    "num_subjects": len(subjects),
                    "num_topics": len(topics),
                    "num_difficulties": len(difficulties)
                }
            }
            
            with open(self.output_dir / "model_config.json", 'w') as f:
                json.dump(model_config, f, indent=2)
            
            logger.info(f"✅ Simplified model saved to: {model_save_path}")
            
            # Evaluate on test set
            logger.info("🧪 Evaluating on test set...")
            
            # Simple evaluation without predict method
            model.eval()
            correct_subjects = 0
            correct_topics = 0
            correct_difficulties = 0
            total_samples = 0
            
            with torch.no_grad():
                for i in range(len(test_dataset)):
                    sample = test_dataset[i]
                    inputs = {
                        "input_ids": sample["input_ids"].unsqueeze(0),
                        "attention_mask": sample["attention_mask"].unsqueeze(0)
                    }
                    
                    outputs = model(**inputs)
                    
                    # Get predictions
                    subject_pred = torch.argmax(outputs["subject_logits"], dim=1).item()
                    topic_pred = torch.argmax(outputs["topic_logits"], dim=1).item()
                    difficulty_pred = torch.argmax(outputs["difficulty_logits"], dim=1).item()
                    
                    # Get true labels
                    subject_true = sample["labels"]["subject"].item()
                    topic_true = sample["labels"]["topic"].item()
                    difficulty_true = sample["labels"]["difficulty"].item()
                    
                    # Check accuracy
                    if subject_pred == subject_true:
                        correct_subjects += 1
                    if topic_pred == topic_true:
                        correct_topics += 1
                    if difficulty_pred == difficulty_true:
                        correct_difficulties += 1
                    
                    total_samples += 1
            
            # Calculate accuracies
            subject_accuracy = correct_subjects / total_samples
            topic_accuracy = correct_topics / total_samples
            difficulty_accuracy = correct_difficulties / total_samples
            
            logger.info(f"📊 Test Set Evaluation Results:")
            logger.info(f"  Subject Accuracy: {subject_accuracy:.3f} ({subject_accuracy*100:.1f}%)")
            logger.info(f"  Topic Accuracy: {topic_accuracy:.3f} ({topic_accuracy*100:.1f}%)")
            logger.info(f"  Difficulty Accuracy: {difficulty_accuracy:.3f} ({difficulty_accuracy*100:.1f}%)")
            
            # Store results
            results = {
                "training_timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_name": "SimpleMultiTaskModel",
                    "base_model": model_name,
                    "training_strategy": "simplified_multi_task",
                    "enhanced_dataset": True,
                    "total_exams": len(training_data["stats"]["exam_types"]),
                    "total_subjects": len(training_data["stats"]["subjects"]),
                    "total_topics": len(training_data["stats"]["topics"])
                },
                "dataset_stats": {
                    "total_questions": training_data["total_questions"],
                    "train_size": len(training_data["train"]),
                    "validation_size": len(training_data["validation"]),
                    "test_size": len(training_data["test"]),
                    "exam_types": training_data["stats"]["exam_types"],
                    "subjects": training_data["stats"]["subjects"],
                    "topics": len(training_data["stats"]["topics"]),
                    "difficulties": training_data["stats"]["difficulties"]
                },
                "performance_metrics": {
                    "subject_accuracy": subject_accuracy,
                    "topic_accuracy": topic_accuracy,
                    "difficulty_accuracy": difficulty_accuracy,
                    "average_accuracy": (subject_accuracy + topic_accuracy + difficulty_accuracy) / 3
                },
                "model_path": str(model_save_path),
                "config_path": str(self.output_dir / "model_config.json")
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"error": str(e)}

    def generate_report(self, training_data: Dict[str, Any], training_results: Dict[str, Any]) -> str:
        """Generate training report"""
        
        report = f"""# 🚀 Enhanced Government Exam AI Model - Training Report

## 📋 Executive Summary

The custom transformer model has been successfully enhanced and trained on a comprehensive dataset covering **{len(training_data['stats']['exam_types'])} major Indian government exams** with **{training_data['total_questions']} questions** across multiple subjects and domains.

---

## 🎯 Training Overview

### Dataset Enhancement
- **Original Dataset**: {training_data.get('original_questions', 0)} questions from basic exams
- **Enhanced Dataset**: {training_data['enhanced_questions']} questions from 15 exams  
- **Combined Total**: {training_data['total_questions']} questions

### Exam Coverage
{chr(10).join([f"- **{exam}**" for exam in training_data['stats']['exam_types']])}

---

## 🏆 Model Performance

### Multi-Task Classification Results
- **Subject Classification**: {training_results['performance_metrics']['subject_accuracy']:.1%} accuracy
- **Topic Classification**: {training_results['performance_metrics']['topic_accuracy']:.1%} accuracy  
- **Difficulty Assessment**: {training_results['performance_metrics']['difficulty_accuracy']:.1%} accuracy
- **Overall Average**: {training_results['performance_metrics']['average_accuracy']:.1%} accuracy

---

## 📊 Coverage Analysis

### Subject Distribution
{chr(10).join([f"- **{subject}**" for subject in training_data['stats']['subjects']])}

### Difficulty Levels
{chr(10).join([f"- **{level}**" for level in training_data['stats']['difficulties']])}

---

## 🚀 Deployment Readiness

### Model Artifacts
- **Trained Model**: `{training_results['model_path']}`
- **Configuration**: `{training_results['config_path']}`

### API Integration
The enhanced model supports **15 government exam types** with real-time inference capabilities.

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Enhanced AI Government Exam System*
"""
        
        return report

    def run_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        
        logger.info("🎯 Starting Simplified Enhanced Training Pipeline...")
        
        # Step 1: Prepare training data
        training_data = self.prepare_training_data()
        
        # Step 2: Train model
        training_results = self.train_simplified_model(training_data)
        
        if "error" in training_results:
            logger.error(f"Training failed: {training_results['error']}")
            return training_results
        
        # Step 3: Generate report
        report = self.generate_report(training_data, training_results)
        
        # Save report
        report_path = self.output_dir / "training_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Training completed!")
        logger.info(f"📄 Report saved to: {report_path}")
        logger.info(f"📊 Model performance: {training_results['performance_metrics']['average_accuracy']:.1%} average accuracy")
        
        # Save final results
        final_results = {
            "pipeline_completion": datetime.now().isoformat(),
            "training_data": training_data,
            "training_results": training_results,
            "report_path": str(report_path),
            "model_artifacts": {
                "model_path": training_results['model_path'],
                "config_path": training_results['config_path'],
                "training_data_path": str(self.output_dir / "training_data.json")
            }
        }
        
        final_results_path = self.output_dir / "pipeline_results.json"
        with open(final_results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"💾 Final results saved to: {final_results_path}")
        
        return final_results

def main():
    """Main execution function"""
    pipeline = SimplifiedTrainingPipeline()
    results = pipeline.run_training()
    return results

if __name__ == "__main__":
    main()