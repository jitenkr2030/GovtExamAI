"""
Custom Transformer Model Trainer for Indian Government Exams
Author: MiniMax Agent
Date: 2025-12-01

This module provides custom transformer model training capabilities specifically
designed for Indian government exam question classification and answer evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExamDataset(Dataset):
    """Custom dataset for government exam data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, task_type='classification'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.labels is not None:
            if self.task_type == 'classification':
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            elif self.task_type == 'regression':
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item


class GovernmentExamTransformer(nn.Module):
    """
    Custom Transformer model for government exam analysis.
    Supports multi-task learning for question classification and answer evaluation.
    """
    
    def __init__(self, model_name='microsoft/DialoGPT-medium', num_subjects=50, num_topics=200, num_difficulty=3, dropout=0.3):
        super(GovernmentExamTransformer, self).__init__()
        
        # Load base transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze base model parameters for faster training
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Classification heads
        hidden_size = self.config.hidden_size
        
        # Subject classification head
        self.subject_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_subjects)
        )
        
        # Topic classification head  
        self.topic_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_topics)
        )
        
        # Difficulty classification head
        self.difficulty_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_difficulty)
        )
        
        # Answer scoring head (regression)
        self.answer_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Multi-task loss weights
        self.subject_weight = 1.0
        self.topic_weight = 1.0
        self.difficulty_weight = 1.0
        self.scoring_weight = 0.5
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different output formats
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            # Use mean pooling of last hidden state
            import torch
            last_hidden_state = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        
        # Multi-task predictions
        subject_logits = self.subject_classifier(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        difficulty_logits = self.difficulty_classifier(pooled_output)
        score = self.answer_scorer(pooled_output).squeeze(-1)
        
        return {
            'subject_logits': subject_logits,
            'topic_logits': topic_logits,
            'difficulty_logits': difficulty_logits,
            'score': score,
            'pooled_output': pooled_output
        }


class MultiTaskLoss(nn.Module):
    """Custom multi-task loss function."""
    
    def __init__(self, subject_weight=1.0, topic_weight=1.0, difficulty_weight=1.0, scoring_weight=0.5):
        super(MultiTaskLoss, self).__init__()
        self.subject_weight = subject_weight
        self.topic_weight = topic_weight
        self.difficulty_weight = difficulty_weight
        self.scoring_weight = scoring_weight
        self.subject_loss = nn.CrossEntropyLoss()
        self.topic_loss = nn.CrossEntropyLoss()
        self.difficulty_loss = nn.CrossEntropyLoss()
        self.scoring_loss = nn.MSELoss()
        
    def forward(self, outputs, labels):
        subject_loss = self.subject_loss(outputs['subject_logits'], labels['subject'])
        topic_loss = self.topic_loss(outputs['topic_logits'], labels['topic'])
        difficulty_loss = self.difficulty_loss(outputs['difficulty_logits'], labels['difficulty'])
        scoring_loss = self.scoring_loss(outputs['score'], labels['score'])
        
        total_loss = (
            self.subject_weight * subject_loss +
            self.topic_weight * topic_loss +
            self.difficulty_weight * difficulty_loss +
            self.scoring_weight * scoring_loss
        )
        
        return total_loss, {
            'total': total_loss,
            'subject': subject_loss,
            'topic': topic_loss,
            'difficulty': difficulty_loss,
            'scoring': scoring_loss
        }


class ModelTrainer:
    """
    Main trainer class for government exam transformer models.
    """
    
    def __init__(self, model_name='microsoft/DialoGPT-medium', config_path='config/exam_categories.py'):
        self.model_name = model_name
        self.config_path = config_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        
        # Load exam categories configuration
        self._load_exam_config()
        
        logger.info(f"Model trainer initialized with device: {self.device}")
        
    def _load_exam_config(self):
        """Load exam categories and configurations."""
        try:
            from config.exam_categories import EXAM_CATEGORIES
            self.exam_config = EXAM_CATEGORIES
            logger.info("Exam categories configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load exam config: {e}")
            self.exam_config = self._create_default_config()
            
    def _create_default_config(self):
        """Create default exam configuration if config file is missing."""
        return {
            'subjects': [f'Subject_{i}' for i in range(50)],
            'topics': [f'Topic_{i}' for i in range(200)],
            'difficulty_levels': ['Easy', 'Medium', 'Hard']
        }
    
    def initialize_model(self, dropout=0.3):
        """Initialize the transformer model with custom heads."""
        logger.info(f"Initializing model with base: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        num_subjects = len(self.exam_config.get('subjects', []))
        num_topics = len(self.exam_config.get('topics', []))
        num_difficulty = len(self.exam_config.get('difficulty_levels', []))
        
        self.model = GovernmentExamTransformer(
            model_name=self.model_name,
            num_subjects=num_subjects,
            num_topics=num_topics,
            num_difficulty=num_difficulty,
            dropout=dropout
        ).to(self.device)
        
        logger.info(f"Model initialized with {num_subjects} subjects, {num_topics} topics, {num_difficulty} difficulty levels")
        
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from JSON, CSV, or pickle file."""
        logger.info(f"Loading training data from {data_path}")
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")
            
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.pkl':
            df = pd.read_pickle(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
        logger.info(f"Loaded {len(df)} training samples")
        return df
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the training data."""
        logger.info("Preprocessing training data")
        
        # Required columns
        required_cols = ['question_text']
        optional_cols = ['subject', 'topic', 'difficulty', 'score', 'answer_text']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from training data")
                
        # Clean question text
        df['question_text'] = df['question_text'].astype(str).str.strip()
        df = df[df['question_text'].str.len() > 10]  # Remove very short questions
        
        # Create combined text for processing
        df['processed_text'] = df['question_text']
        
        # Add answer text if available
        if 'answer_text' in df.columns:
            df['answer_text'] = df['answer_text'].fillna('')
            df['processed_text'] = df['question_text'] + ' [SEP] ' + df['answer_text']
            
        # Encode categorical labels
        if 'subject' in df.columns:
            df['subject_encoded'] = pd.Categorical(df['subject']).codes
            
        if 'topic' in df.columns:
            df['topic_encoded'] = pd.Categorical(df['topic']).codes
            
        if 'difficulty' in df.columns:
            difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
            df['difficulty_encoded'] = df['difficulty'].map(difficulty_map)
            
        # Normalize scores to 0-1 range
        if 'score' in df.columns:
            df['score'] = df['score'].astype(float)
            score_min, score_max = df['score'].min(), df['score'].max()
            if score_max > score_min:
                df['score_normalized'] = (df['score'] - score_min) / (score_max - score_min)
            else:
                df['score_normalized'] = 0.5
                
        logger.info(f"Preprocessed data: {len(df)} valid samples")
        return df
        
    def create_data_loaders(self, df: pd.DataFrame, test_size=0.2, batch_size=16, max_length=512):
        """Create training and validation data loaders."""
        logger.info("Creating data loaders")
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Prepare data for classification task
        train_dataset = ExamDataset(
            train_df['processed_text'].tolist(),
            train_df.get('subject_encoded', [None] * len(train_df)).tolist(),
            self.tokenizer,
            max_length=max_length,
            task_type='classification'
        )
        
        val_dataset = ExamDataset(
            val_df['processed_text'].tolist(),
            val_df.get('subject_encoded', [None] * len(val_df)).tolist(),
            self.tokenizer,
            max_length=max_length,
            task_type='classification'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
        
    def train_classification_model(self, train_loader, val_loader, epochs=3, learning_rate=1e-4):
        """Train the classification model."""
        logger.info("Starting classification model training")
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
            
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        training_history = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['subject_logits'], labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['subject_logits'].data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            # Validation
            val_accuracy, val_loss = self._evaluate_classification_model(val_loader)
            
            epoch_accuracy = correct_predictions / total_predictions
            avg_loss = epoch_loss / len(train_loader)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_accuracy': epoch_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
        self.training_history = training_history
        logger.info("Classification training completed")
        return training_history
        
    def _evaluate_classification_model(self, val_loader):
        """Evaluate classification model."""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['subject_logits'], labels)
                
                _, predicted = torch.max(outputs['subject_logits'].data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                total_loss += loss.item()
                
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(val_loader)
        
        return accuracy, avg_loss
        
    def fine_tune_with_pdfs(self, pdf_dir: str, sample_size: int = 1000):
        """Fine-tune model using PDF data from exam papers."""
        logger.info(f"Fine-tuning with PDF data from {pdf_dir}")
        
        # This would integrate with PDF processing pipeline
        # For now, placeholder implementation
        logger.info("PDF fine-tuning feature - to be implemented with PDF pipeline")
        
    def save_model(self, save_dir: str):
        """Save the trained model and tokenizer."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_dir}")
        
        # Save model state dict
        torch.save(self.model.state_dict(), save_dir / 'model_state.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir / 'tokenizer')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
        # Save configuration
        config = {
            'model_name': self.model_name,
            'exam_config': self.exam_config,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Model saved successfully to {save_dir}")
        
    def load_model(self, save_dir: str):
        """Load a previously saved model."""
        save_dir = Path(save_dir)
        
        logger.info(f"Loading model from {save_dir}")
        
        # Load configuration
        with open(save_dir / 'config.json', 'r') as f:
            config = json.load(f)
            
        self.model_name = config['model_name']
        self.exam_config = config['exam_config']
        
        # Initialize tokenizer and model
        self.initialize_model()
        
        # Load model state
        self.model.load_state_dict(torch.load(save_dir / 'model_state.pt', map_location=self.device))
        self.model.eval()
        
        # Load training history
        try:
            with open(save_dir / 'training_history.json', 'r') as f:
                self.training_history = json.load(f)
        except FileNotFoundError:
            self.training_history = []
            
        logger.info(f"Model loaded successfully from {save_dir}")
        
    def predict(self, text: str, return_all_predictions=False):
        """Make predictions on new text."""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
        # Convert predictions to numpy
        subject_logits = outputs['subject_logits'].cpu().numpy()
        topic_logits = outputs['topic_logits'].cpu().numpy()
        difficulty_logits = outputs['difficulty_logits'].cpu().numpy()
        score = outputs['score'].cpu().numpy()
        
        # Get predicted classes
        subject_pred = np.argmax(subject_logits[0])
        topic_pred = np.argmax(topic_logits[0])
        difficulty_pred = np.argmax(difficulty_logits[0])
        
        # Map back to original labels
        subject = self.exam_config['subjects'][subject_pred] if subject_pred < len(self.exam_config['subjects']) else 'Unknown'
        topic = self.exam_config['topics'][topic_pred] if topic_pred < len(self.exam_config['topics']) else 'Unknown'
        difficulty = self.exam_config['difficulty_levels'][difficulty_pred] if difficulty_pred < len(self.exam_config['difficulty_levels']) else 'Unknown'
        
        result = {
            'subject': subject,
            'topic': topic,
            'difficulty': difficulty,
            'confidence': {
                'subject': float(np.max(subject_logits[0])),
                'topic': float(np.max(topic_logits[0])),
                'difficulty': float(np.max(difficulty_logits[0]))
            }
        }
        
        if return_all_predictions:
            result['all_predictions'] = {
                'subject_probs': subject_logits[0].tolist(),
                'topic_probs': topic_logits[0].tolist(),
                'difficulty_probs': difficulty_logits[0].tolist(),
                'score': float(score[0])
            }
            
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Create sample training data
    sample_data = pd.DataFrame({
        'question_text': [
            'What is the capital of India?',
            'Calculate the compound interest for Rs. 10,000 at 5% for 2 years.',
            'Explain the process of photosynthesis.',
            'What are the fundamental rights in the Indian Constitution?'
        ] * 250,
        'subject': ['Geography', 'Mathematics', 'Biology', 'Polity'] * 250,
        'topic': ['Countries', 'Interest', 'Plant Biology', 'Fundamental Rights'] * 250,
        'difficulty': ['Easy', 'Medium', 'Medium', 'Hard'] * 250
    })
    
    print("Sample data created for testing")
    print(f"Data shape: {sample_data.shape}")
    print(sample_data.head())