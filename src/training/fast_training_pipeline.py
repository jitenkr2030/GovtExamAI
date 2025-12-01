#!/usr/bin/env python3
"""
Fast Training Pipeline for 15 Government Exams
Quick training with reduced epochs and smaller model
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
from sklearn.metrics import accuracy_score
import logging
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastExamClassifier(nn.Module):
    """Fast multi-task classifier with frozen base model"""
    
    def __init__(self, model_name: str, num_subjects: int, num_topics: int, num_difficulties: int):
        super().__init__()
        
        from transformers import DistilBertModel, AutoTokenizer
        
        # Load base model
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze all base model parameters for faster training
        for param in self.distilbert.parameters():
            param.requires_grad = False
            
        hidden_size = self.distilbert.config.hidden_size
        
        # Simple linear classifiers
        self.subject_classifier = nn.Linear(hidden_size, num_subjects)
        self.topic_classifier = nn.Linear(hidden_size, num_topics)
        self.difficulty_classifier = nn.Linear(hidden_size, num_difficulties)
        
    def forward(self, input_ids=None, attention_mask=None):
        # Base model forward pass
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
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
        
        # Combined loss
        total_loss = subject_loss + topic_loss + difficulty_loss
        
        return total_loss

class FastExamDataset(Dataset):
    """Fast dataset for exam questions"""
    
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
            max_length=256,  # Reduced sequence length
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "subject_labels": torch.tensor(self.subject_encoder.transform([question["target"]["subject"]])[0], dtype=torch.long),
            "topic_labels": torch.tensor(self.topic_encoder.transform([question["target"]["topic"]])[0], dtype=torch.long),
            "difficulty_labels": torch.tensor(self.difficulty_encoder.transform([question["target"]["difficulty"]])[0], dtype=torch.long)
        }

def main():
    """Fast training and evaluation"""
    
    logger.info("🚀 Starting Fast Government Exam AI Training")
    logger.info("=" * 60)
    
    # Initialize paths
    base_dir = Path("/workspace")
    output_dir = base_dir / "fast_training_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Load enhanced dataset
    dataset_path = base_dir / "data_collection/enhanced_exam_data/enhanced_exam_dataset.json"
    logger.info(f"Loading enhanced dataset from: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        enhanced_dataset = json.load(f)
    
    logger.info(f"Loaded {enhanced_dataset['total_questions']} questions from {enhanced_dataset['total_exam_types']} exam types")
    
    # Load original dataset
    original_dataset_path = base_dir / "training_outputs/processed_training_data.json"
    if original_dataset_path.exists():
        with open(original_dataset_path, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
        logger.info("Loaded original dataset for comparison")
    else:
        original_dataset = None
    
    # Prepare all questions
    all_questions = []
    
    # Add enhanced synthetic questions
    for q in enhanced_dataset["questions"]:
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
    if original_dataset:
        for split in ["train", "validation", "test"]:
            if split in original_dataset:
                for q in original_dataset[split]:
                    q["metadata"]["data_type"] = "original_real"
                    all_questions.append(q)
    
    logger.info(f"Combined dataset: {len(all_questions)} total questions")
    
    # Create splits
    train_questions, temp_questions = train_test_split(all_questions, test_size=0.2, random_state=42)
    val_questions, test_questions = train_test_split(temp_questions, test_size=0.5, random_state=42)
    
    logger.info(f"Train: {len(train_questions)}, Val: {len(val_questions)}, Test: {len(test_questions)}")
    
    # Extract unique labels
    all_data = train_questions + val_questions + test_questions
    
    subjects = list(set(q["target"]["subject"] for q in all_data))
    topics = list(set(q["target"]["topic"] for q in all_data))
    difficulties = list(set(q["target"]["difficulty"] for q in all_data))
    
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
    train_dataset = FastExamDataset(train_questions, tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
    val_dataset = FastExamDataset(val_questions, tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
    test_dataset = FastExamDataset(test_questions, tokenizer, subject_encoder, topic_encoder, difficulty_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Larger batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = FastExamClassifier(
        model_name=model_name,
        num_subjects=len(subjects),
        num_topics=len(topics),
        num_difficulties=len(difficulties)
    )
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Training settings - FASTER TRAINING
    num_epochs = 2  # Reduced epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Fast training for {num_epochs} epochs with batch size 16")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subject_labels = batch["subject_labels"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            difficulty_labels = batch["difficulty_labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = model.compute_loss(
                outputs["subject_logits"], outputs["topic_logits"], outputs["difficulty_logits"],
                subject_labels, topic_labels, difficulty_labels
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
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
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    model.eval()
    
    logger.info("📊 Evaluating model on test data...")
    
    # Evaluate on test set
    all_subject_preds = []
    all_topic_preds = []
    all_difficulty_preds = []
    all_subject_true = []
    all_topic_true = []
    all_difficulty_true = []
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subject_labels = batch["subject_labels"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            difficulty_labels = batch["difficulty_labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            subject_preds = torch.argmax(outputs["subject_logits"], dim=-1)
            topic_preds = torch.argmax(outputs["topic_logits"], dim=-1)
            difficulty_preds = torch.argmax(outputs["difficulty_logits"], dim=-1)
            
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
    
    # Save results
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
    
    results = {
        "subject_accuracy": float(subject_accuracy),
        "topic_accuracy": float(topic_accuracy),
        "difficulty_accuracy": float(difficulty_accuracy),
        "overall_accuracy": float(overall_accuracy),
        "test_samples": len(test_dataset)
    }
    
    # Save all outputs
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    logger.info("🎉 Fast training completed successfully!")
    logger.info("=" * 60)
    logger.info("📋 FINAL RESULTS SUMMARY:")
    logger.info(f"Total Questions: {len(all_questions)}")
    logger.info(f"Exam Types: {enhanced_dataset['total_exam_types']}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Topics: {len(topics)}")
    logger.info(f"Difficulties: {len(difficulties)}")
    logger.info("")
    logger.info("🎯 ACCURACY RESULTS:")
    logger.info(f"Subject Classification: {subject_accuracy:.4f}")
    logger.info(f"Topic Classification: {topic_accuracy:.4f}")
    logger.info(f"Difficulty Classification: {difficulty_accuracy:.4f}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info("")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info(f"Results saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    main()