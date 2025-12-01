#!/usr/bin/env python3
"""
Comprehensive evaluation of the trained government exam AI model
Evaluates performance on all 15 government exams with detailed metrics
"""

import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, DistilBertModel
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectExamClassifier(nn.Module):
    def __init__(self, model_name, num_subjects, num_topics, num_difficulties):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Three classification heads
        self.subject_classifier = nn.Linear(768, num_subjects)
        self.topic_classifier = nn.Linear(768, num_topics)
        self.difficulty_classifier = nn.Linear(768, num_difficulties)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        subject_logits = self.subject_classifier(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        difficulty_logits = self.difficulty_classifier(pooled_output)
        
        return subject_logits, topic_logits, difficulty_logits

class EvaluationDataset:
    def __init__(self, data_path, tokenizer, subject_encoder, topic_encoder, difficulty_encoder):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.subject_encoder = subject_encoder
        self.topic_encoder = topic_encoder
        self.difficulty_encoder = difficulty_encoder
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize question text
        encoding = self.tokenizer(
            item['question'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Encode labels
        subject_label = self.subject_encoder.transform([item['subject']])[0]
        topic_label = self.topic_encoder.transform([item['topic']])[0]
        difficulty_label = self.difficulty_encoder.transform([item['difficulty']])[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'subject_labels': torch.tensor(subject_label, dtype=torch.long),
            'topic_labels': torch.tensor(topic_label, dtype=torch.long),
            'difficulty_labels': torch.tensor(difficulty_label, dtype=torch.long),
            'exam_type': item['exam_type'],
            'question': item['question']
        }

def load_and_prepare_data():
    """Load enhanced dataset and prepare encoders"""
    with open('/workspace/data_collection/enhanced_exam_data/enhanced_exam_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract unique labels
    subjects = sorted(list(set(item['subject'] for item in data)))
    topics = sorted(list(set(item['topic'] for item in data)))
    difficulties = sorted(list(set(item['difficulty'] for item in data)))
    
    # Create encoders
    subject_encoder = LabelEncoder()
    topic_encoder = LabelEncoder()
    difficulty_encoder = LabelEncoder()
    
    subject_encoder.fit(subjects)
    topic_encoder.fit(topics)
    difficulty_encoder.fit(difficulties)
    
    logger.info(f"Dataset loaded: {len(data)} questions")
    logger.info(f"Subjects: {len(subjects)}, Topics: {len(topics)}, Difficulties: {len(difficulties)}")
    
    return data, subject_encoder, topic_encoder, difficulty_encoder, subjects, topics, difficulties

def evaluate_model():
    """Comprehensive evaluation of the trained model"""
    logger.info("Starting model evaluation...")
    
    # Load data and prepare encoders
    data, subject_encoder, topic_encoder, difficulty_encoder, subjects, topics, difficulties = load_and_prepare_data()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model_name = 'distilbert-base-uncased'
    model = DirectExamClassifier(
        model_name, 
        len(subjects), 
        len(topics), 
        len(difficulties)
    )
    
    # Load trained weights
    model_path = '/workspace/direct_training_outputs/best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded trained model from {model_path}")
    else:
        logger.error(f"Model file not found: {model_path}")
        return
    
    model.to(device)
    model.eval()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create evaluation dataset
    eval_dataset = EvaluationDataset(
        '/workspace/data_collection/enhanced_exam_data/enhanced_exam_dataset.json',
        tokenizer,
        subject_encoder,
        topic_encoder,
        difficulty_encoder
    )
    
    # Create data loader
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    
    # Evaluation
    all_predictions = {'subject': [], 'topic': [], 'difficulty': []}
    all_labels = {'subject': [], 'topic': [], 'difficulty': []}
    exam_wise_performance = {}
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            subject_labels = batch['subject_labels'].to(device)
            topic_labels = batch['topic_labels'].to(device)
            difficulty_labels = batch['difficulty_labels'].to(device)
            
            # Forward pass
            subject_logits, topic_logits, difficulty_logits = model(input_ids, attention_mask)
            
            # Get predictions
            subject_preds = torch.argmax(subject_logits, dim=1)
            topic_preds = torch.argmax(topic_logits, dim=1)
            difficulty_preds = torch.argmax(difficulty_logits, dim=1)
            
            # Store predictions and labels
            all_predictions['subject'].extend(subject_preds.cpu().numpy())
            all_predictions['topic'].extend(topic_preds.cpu().numpy())
            all_predictions['difficulty'].extend(difficulty_preds.cpu().numpy())
            
            all_labels['subject'].extend(subject_labels.cpu().numpy())
            all_labels['topic'].extend(topic_labels.cpu().numpy())
            all_labels['difficulty'].extend(difficulty_labels.cpu().numpy())
            
            # Store exam-wise performance
            for i, exam_type in enumerate(batch['exam_type']):
                if exam_type not in exam_wise_performance:
                    exam_wise_performance[exam_type] = {
                        'subject_preds': [], 'subject_true': [],
                        'topic_preds': [], 'topic_true': [],
                        'difficulty_preds': [], 'difficulty_true': []
                    }
                
                exam_wise_performance[exam_type]['subject_preds'].append(subject_preds[i].cpu().numpy())
                exam_wise_performance[exam_type]['subject_true'].append(subject_labels[i].cpu().numpy())
                exam_wise_performance[exam_type]['topic_preds'].append(topic_preds[i].cpu().numpy())
                exam_wise_performance[exam_type]['topic_true'].append(topic_labels[i].cpu().numpy())
                exam_wise_performance[exam_type]['difficulty_preds'].append(difficulty_preds[i].cpu().numpy())
                exam_wise_performance[exam_type]['difficulty_true'].append(difficulty_labels[i].cpu().numpy())
    
    # Calculate overall metrics
    overall_results = {}
    for task in ['subject', 'topic', 'difficulty']:
        accuracy = accuracy_score(all_labels[task], all_predictions[task])
        overall_results[task] = accuracy
        logger.info(f"{task.capitalize()} Accuracy: {accuracy:.4f}")
    
    # Generate detailed report
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_questions': len(data),
            'total_exams': len(set(item['exam_type'] for item in data)),
            'subjects': len(subjects),
            'topics': len(topics),
            'difficulties': len(difficulties)
        },
        'overall_accuracy': overall_results,
        'exam_wise_performance': {},
        'exam_types': list(set(item['exam_type'] for item in data))
    }
    
    # Calculate exam-wise performance
    for exam_type, perf in exam_wise_performance.items():
        exam_results = {}
        for task in ['subject', 'topic', 'difficulty']:
            task_preds = perf[f'{task}_preds']
            task_true = perf[f'{task}_true']
            if task_preds:  # Check if there are predictions
                accuracy = accuracy_score(task_true, task_preds)
                exam_results[f'{task}_accuracy'] = accuracy
        
        # Overall accuracy for this exam
        if exam_results:
            overall_acc = np.mean(list(exam_results.values()))
            exam_results['overall_accuracy'] = overall_acc
            exam_results['sample_count'] = len(perf['subject_preds'])
        
        report['exam_wise_performance'][exam_type] = exam_results
    
    # Save detailed report
    report_path = '/workspace/training_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation completed. Report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("GOVERNMENT EXAM AI MODEL - EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions Evaluated: {len(data)}")
    print(f"Total Exam Types: {len(set(item['exam_type'] for item in data))}")
    print(f"Overall Subject Accuracy: {overall_results['subject']:.2%}")
    print(f"Overall Topic Accuracy: {overall_results['topic']:.2%}")
    print(f"Overall Difficulty Accuracy: {overall_results['difficulty']:.2%}")
    print("\nTop Performing Exams:")
    
    # Sort exams by overall accuracy
    sorted_exams = sorted(report['exam_wise_performance'].items(), 
                         key=lambda x: x[1].get('overall_accuracy', 0), reverse=True)
    
    for exam_type, results in sorted_exams[:10]:
        if 'overall_accuracy' in results:
            print(f"  {exam_type}: {results['overall_accuracy']:.2%} ({results.get('sample_count', 0)} samples)")
    
    return report

if __name__ == "__main__":
    evaluate_model()