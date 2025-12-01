#!/usr/bin/env python3
"""
Direct PyTorch Training Pipeline for 15 Government Exams
Bypasses Trainer compatibility issues by using pure PyTorch
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectExamClassifier(nn.Module):
    """Direct PyTorch multi-task classifier for government exams"""
    
    def __init__(self, model_name: str, num_subjects: int, num_topics: int, num_difficulties: int):
        super().__init__()
        
        from transformers import DistilBertModel, AutoTokenizer
        
        # Load base model
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze base model parameters for faster training
        for param in self.distilbert.parameters():
            param.requires_grad = False
            
        hidden_size = self.distilbert.config.hidden_size
        
        # Multi-task classification heads
        self.subject_classifier = nn.Linear(hidden_size, num_subjects)
        self.topic_classifier = nn.Linear(hidden_size, num_topics)
        self.difficulty_classifier = nn.Linear(hidden_size, num_difficulties)
        
    def forward(self, input_ids=None, attention_mask=None):
        # Base model forward pass
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Get predictions from each classifier
        subject_logits = self.subject_classifier(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        difficulty_logits = self.difficulty_classifier(pooled_output)
        
        return {
            "subject_logits": subject_logits,
            "topic_logits": topic_logits,
            "difficulty_logits": difficulty_logits
        }
        
    def compute_loss(self, subject_logits, topic_logits, difficulty_logits, 
                    subject_labels, topic_labels, difficulty_labels):
        """Compute combined loss"""
        subject_loss_fct = nn.CrossEntropyLoss()
        topic_loss_fct = nn.CrossEntropyLoss()
        difficulty_loss_fct = nn.CrossEntropyLoss()
        
        subject_loss = subject_loss_fct(subject_logits, subject_labels)
        topic_loss = topic_loss_fct(topic_logits, topic_labels)
        difficulty_loss = difficulty_loss_fct(difficulty_logits, difficulty_labels)
        
        # Combined loss with equal weights
        total_loss = subject_loss + topic_loss + difficulty_loss
        
        return total_loss

class DirectExamDataset(Dataset):
    """Direct PyTorch dataset for exam questions"""
    
    def __init__(self, questions: List[Dict], tokenizer, subject_encoder, topic_encoder, difficulty_encoder):
        self.questions = questions
        self.tokenizer = tokenizer
        self.subject_encoder = subject_encoder
        self.topic_encoder = topic_encoder
        self.difficulty_encoder = difficulty_encoder
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        question = self.questions[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            question["input_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "subject_labels": torch.tensor(self.subject_encoder.transform([question["target"]["subject"]])[0], dtype=torch.long),
            "topic_labels": torch.tensor(self.topic_encoder.transform([question["target"]["topic"]])[0], dtype=torch.long),
            "difficulty_labels": torch.tensor(self.difficulty_encoder.transform([question["target"]["difficulty"]])[0], dtype=torch.long)
        }

class DirectTrainingPipeline:
    """Direct PyTorch training pipeline"""
    
    def __init__(self, base_dir="/workspace"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "direct_training_outputs"
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
                        q["metadata"]["data_type"] = "original_real"
                        all_questions.append(q)
        
        logger.info(f"Combined dataset: {len(all_questions)} total questions")
        
        # Create simple random splits (80/10/10)
        train_questions, temp_questions = train_test_split(
            all_questions, 
            test_size=0.2, 
            random_state=42
        )
        
        val_questions, test_questions = train_test_split(
            temp_questions,
            test_size=0.5,  # 10% of total for each val and test
            random_state=42
        )
        
        return {
            "train": train_questions,
            "validation": val_questions,
            "test": test_questions,
            "total": len(all_questions),
            "train_count": len(train_questions),
            "val_count": len(val_questions),
            "test_count": len(test_questions)
        }

    def train_direct_model(self) -> Dict[str, Any]:
        """Train model using direct PyTorch"""
        
        logger.info("🚀 Starting direct PyTorch training pipeline...")
        
        # Prepare data
        splits = self.prepare_training_data()
        
        # Extract unique labels
        all_questions = splits["train"] + splits["validation"] + splits["test"]
        
        subjects = list(set(q["target"]["subject"] for q in all_questions))
        topics = list(set(q["target"]["topic"] for q in all_questions))
        difficulties = list(set(q["target"]["difficulty"] for q in all_questions))
        
        logger.info(f"Found {len(subjects)} subjects, {len(topics)} topics, {len(difficulties)} difficulties")
        
        # Create encoders
        subject_encoder = LabelEncoder()
        topic_encoder = LabelEncoder()
        difficulty_encoder = LabelEncoder()
        
        subject_encoder.fit(subjects)
        topic_encoder.fit(topics)
        difficulty_encoder.fit(difficulties)
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create datasets
        train_dataset = DirectExamDataset(splits["train"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
        val_dataset = DirectExamDataset(splits["validation"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
        test_dataset = DirectExamDataset(splits["test"], tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Initialize model
        model = DirectExamClassifier(
            model_name=model_name,
            num_subjects=len(subjects),
            num_topics=len(topics),
            num_difficulties=len(difficulties)
        )
        
        # Initialize optimizer and loss
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Training settings
        num_epochs = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"Using device: {device}")
        logger.info(f"Training for {num_epochs} epochs")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                subject_labels = batch["subject_labels"].to(device)
                topic_labels = batch["topic_labels"].to(device)
                difficulty_labels = batch["difficulty_labels"].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Compute loss
                loss = model.compute_loss(
                    outputs["subject_logits"], outputs["topic_logits"], outputs["difficulty_logits"],
                    subject_labels, topic_labels, difficulty_labels
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    subject_labels = batch["subject_labels"].to(device)
                    topic_labels = batch["topic_labels"].to(device)
                    difficulty_labels = batch["difficulty_labels"].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    loss = model.compute_loss(
                        outputs["subject_logits"], outputs["topic_logits"], outputs["difficulty_logits"],
                        subject_labels, topic_labels, difficulty_labels
                    )
                    
                    total_val_loss += loss.item()
                    val_batches += 1
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save model state dict
                torch.save(model.state_dict(), self.output_dir / "best_model.pt")
                logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(self.output_dir / "best_model.pt"))
        
        # Save configuration
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
        
        # Save training data
        with open(self.output_dir / "direct_training_data.json", 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"✅ Direct model training completed successfully!")
        
        # Run evaluation
        results = self.evaluate_direct_model(model, test_dataset, device)
        
        return {
            "status": "success",
            "model_path": str(self.output_dir / "best_model.pt"),
            "config_path": str(self.output_dir / "model_config.json"),
            "results": results,
            "summary": {
                "total_questions": splits["total"],
                "train_size": splits["train_count"],
                "val_size": splits["val_count"],
                "test_size": splits["test_count"],
                "subjects": len(subjects),
                "topics": len(topics),
                "difficulties": len(difficulties),
                "exam_types": self.enhanced_dataset['total_exam_types']
            }
        }

    def evaluate_direct_model(self, model, test_dataset, device) -> Dict[str, Any]:
        """Evaluate the trained model on test data"""
        
        logger.info("📊 Evaluating direct model on test data...")
        
        # Load configuration
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create test data loader
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Set model to evaluation mode
        model.eval()
        
        all_subject_preds = []
        all_topic_preds = []
        all_difficulty_preds = []
        all_subject_true = []
        all_topic_true = []
        all_difficulty_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                subject_labels = batch["subject_labels"].to(device)
                topic_labels = batch["topic_labels"].to(device)
                difficulty_labels = batch["difficulty_labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get predictions
                subject_preds = torch.argmax(outputs["subject_logits"], dim=-1)
                topic_preds = torch.argmax(outputs["topic_logits"], dim=-1)
                difficulty_preds = torch.argmax(outputs["difficulty_logits"], dim=-1)
                
                # Store predictions and true labels
                all_subject_preds.extend(subject_preds.cpu().numpy())
                all_topic_preds.extend(topic_preds.cpu().numpy())
                all_difficulty_preds.extend(difficulty_preds.cpu().numpy())
                
                all_subject_true.extend(subject_labels.cpu().numpy())
                all_topic_true.extend(topic_labels.cpu().numpy())
                all_difficulty_true.extend(difficulty_labels.cpu().numpy())
        
        # Calculate accuracies
        subject_accuracy = accuracy_score(all_subject_true, all_subject_preds)
        topic_accuracy = accuracy_score(all_topic_true, all_topic_preds)
        difficulty_accuracy = accuracy_score(all_difficulty_true, all_difficulty_preds)
        
        overall_accuracy = np.mean([subject_accuracy, topic_accuracy, difficulty_accuracy])
        
        results = {
            "subject_accuracy": float(subject_accuracy),
            "topic_accuracy": float(topic_accuracy),
            "difficulty_accuracy": float(difficulty_accuracy),
            "overall_accuracy": float(overall_accuracy),
            "test_samples": len(test_dataset)
        }
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"🎯 Final Results:")
        logger.info(f"Subject Accuracy: {subject_accuracy:.4f}")
        logger.info(f"Topic Accuracy: {topic_accuracy:.4f}")
        logger.info(f"Difficulty Accuracy: {difficulty_accuracy:.4f}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
        
        return results

def main():
    """Main execution function"""
    
    logger.info("🚀 Starting Direct PyTorch Government Exam AI Training Pipeline")
    logger.info("=" * 70)
    
    # Initialize pipeline
    pipeline = DirectTrainingPipeline()
    
    # Train model
    results = pipeline.train_direct_model()
    
    if results["status"] == "success":
        logger.info("🎉 Direct training pipeline completed successfully!")
        logger.info("=" * 70)
        logger.info("📋 FINAL RESULTS SUMMARY:")
        logger.info(f"Total Questions: {results['summary']['total_questions']}")
        logger.info(f"Exam Types: {results['summary']['exam_types']}")
        logger.info(f"Subjects: {results['summary']['subjects']}")
        logger.info(f"Topics: {results['summary']['topics']}")
        logger.info(f"Difficulties: {results['summary']['difficulties']}")
        logger.info("")
        logger.info("🎯 ACCURACY RESULTS:")
        logger.info(f"Subject Classification: {results['results']['subject_accuracy']:.4f}")
        logger.info(f"Topic Classification: {results['results']['topic_accuracy']:.4f}")
        logger.info(f"Difficulty Classification: {results['results']['difficulty_accuracy']:.4f}")
        logger.info(f"Overall Accuracy: {results['results']['overall_accuracy']:.4f}")
        logger.info("")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Results saved to: {pipeline.output_dir}")
    else:
        logger.error("❌ Training pipeline failed")
        
    return results

if __name__ == "__main__":
    main()