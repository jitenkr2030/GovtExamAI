"""
Custom Transformer Training Pipeline for Government Exams
Author: MiniMax Agent
Date: 2025-12-01

This script provides end-to-end training capabilities for custom transformer models
on Indian government exam data, including data collection, preprocessing, training,
and model deployment.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from ml_models.model_trainer import ModelTrainer
from data_ingestion.data_collection_pipeline import DatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GovernmentExamModelPipeline:
    """End-to-end training pipeline for government exam models."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.dataset_builder = DatasetBuilder(self.config.get('dataset_config', {}))
        self.model_trainer = ModelTrainer(
            model_name=self.config.get('model_config', {}).get('base_model', 'microsoft/DialoGPT-medium'),
            config_path=self.config.get('exam_config_path')
        )
        self.training_results = {}
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load training configuration."""
        default_config = {
            'model_config': {
                'base_model': 'microsoft/DialoGPT-medium',
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 1e-4,
                'max_length': 512,
                'dropout': 0.3,
                'save_dir': 'models/government_exam_model'
            },
            'dataset_config': {
                'min_text_length': 20,
                'augmentation_factor': 2,
                'deduplicate': True,
                'test_size': 0.2,
                'validation_size': 0.1
            },
            'data_sources': {
                'pdf_directory': 'data/pdfs',
                'text_files': 'data/text_files',
                'json_files': 'data/json_files',
                'sample_data': 'data/sample_exam_data.json'
            },
            'training_settings': {
                'use_augmentation': True,
                'save_intermediate': True,
                'early_stopping': True,
                'validation_metric': 'accuracy'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
        
    def collect_and_prepare_data(self, sources: Optional[List[str]] = None) -> pd.DataFrame:
        """Collect and prepare training data from multiple sources."""
        logger.info("Starting data collection and preparation")
        
        if sources is None:
            sources = []
            
        # Add configured data sources
        for source_path, source_type in self.config['data_sources'].items():
            if os.path.exists(source_path):
                logger.info(f"Adding data source: {source_path}")
                self.dataset_builder.add_data_source(source_path, source_type)
            else:
                logger.warning(f"Data source not found: {source_path}")
                
        # Add custom sources
        for source in sources:
            if os.path.exists(source):
                ext = Path(source).suffix.lower()
                source_type = self._determine_source_type(source)
                self.dataset_builder.add_data_source(source, source_type)
                
        # Process and clean data
        logger.info("Processing and enhancing collected data")
        processed_df = self.dataset_builder.process_and_enhance_data(
            min_text_length=self.config['dataset_config']['min_text_length'],
            deduplicate=self.config['dataset_config']['deduplicate']
        )
        
        if len(processed_df) == 0:
            raise ValueError("No valid training data collected. Please check data sources.")
            
        # Display data statistics
        stats = self.dataset_builder.dataset_stats
        logger.info(f"Data collection complete:")
        logger.info(f"  Total questions: {stats['total_questions']}")
        logger.info(f"  Sources: {list(stats['sources'].keys())}")
        logger.info(f"  Subjects: {list(stats['subjects'].keys())}")
        logger.info(f"  Difficulty levels: {list(stats['difficulty_levels'].keys())}")
        
        # Augment data if configured
        if self.config['training_settings']['use_augmentation']:
            logger.info(f"Augmenting dataset with factor {self.config['dataset_config']['augmentation_factor']}")
            processed_df = self.dataset_builder.augment_dataset(
                augmentation_factor=self.config['dataset_config']['augmentation_factor']
            )
            
        return processed_df
        
    def _determine_source_type(self, file_path: str) -> str:
        """Determine data source type from file extension."""
        ext = Path(file_path).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'text'
        }
        return type_mapping.get(ext, 'json')
        
    def train_model(self, data: Optional[pd.DataFrame] = None, 
                   save_model: bool = True) -> Dict:
        """Train the custom transformer model."""
        logger.info("Starting model training")
        
        if data is None:
            # Use processed data from dataset builder
            if not self.dataset_builder.processed_data:
                raise ValueError("No processed data available. Run collect_and_prepare_data() first.")
            data = pd.DataFrame(self.dataset_builder.processed_data)
            
        # Prepare training data
        train_df, val_df, test_df = self.dataset_builder.get_training_ready_data(
            test_size=self.config['dataset_config']['test_size'],
            validation_size=self.config['dataset_config']['validation_size']
        )
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Initialize model
        logger.info(f"Initializing model with base: {self.config['model_config']['base_model']}")
        self.model_trainer.initialize_model(dropout=self.config['model_config']['dropout'])
        
        # Create data loaders
        train_loader, val_loader = self.model_trainer.create_data_loaders(
            train_df, val_df, 
            batch_size=self.config['model_config']['batch_size'],
            max_length=self.config['model_config']['max_length']
        )
        
        # Train model
        training_history = self.model_trainer.train_classification_model(
            train_loader, val_loader,
            epochs=self.config['model_config']['epochs'],
            learning_rate=self.config['model_config']['learning_rate']
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set")
        test_loader, _ = self.model_trainer.create_data_loaders(
            test_df, test_df,  # Use test_df for both to create single batch loader
            batch_size=self.config['model_config']['batch_size'],
            max_length=self.config['model_config']['max_length']
        )
        
        test_accuracy, test_loss = self.model_trainer._evaluate_classification_model(test_loader)
        
        # Save model if requested
        if save_model:
            save_dir = Path(self.config['model_config']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model_trainer.save_model(str(save_dir))
            
        # Compile results
        final_results = {
            'training_history': training_history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model_config': self.config['model_config'],
            'dataset_stats': self.dataset_builder.dataset_stats,
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.training_results = final_results
        
        logger.info(f"Training completed!")
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        logger.info(f"Final test loss: {test_loss:.4f}")
        
        return final_results
        
    def fine_tune_existing_model(self, model_path: str, data: pd.DataFrame) -> Dict:
        """Fine-tune an existing model on new data."""
        logger.info(f"Fine-tuning existing model from {model_path}")
        
        # Load existing model
        self.model_trainer.load_model(model_path)
        
        # Use a smaller learning rate for fine-tuning
        fine_tune_config = self.config['model_config'].copy()
        fine_tune_config['learning_rate'] = fine_tune_config['learning_rate'] * 0.1
        fine_tune_config['epochs'] = max(1, fine_tune_config['epochs'] // 2)
        
        logger.info(f"Fine-tuning with learning rate: {fine_tune_config['learning_rate']}")
        logger.info(f"Fine-tuning epochs: {fine_tune_config['epochs']}")
        
        # Train model (same as regular training but with existing model)
        results = self.train_model(data, save_model=False)
        
        # Save fine-tuned model
        save_dir = Path(model_path).parent / f"{Path(model_path).name}_fine_tuned"
        self.model_trainer.save_model(str(save_dir))
        
        results['fine_tuned'] = True
        results['base_model_path'] = model_path
        
        return results
        
    def evaluate_model(self, model_path: str, test_data: pd.DataFrame) -> Dict:
        """Evaluate a trained model on test data."""
        logger.info(f"Evaluating model from {model_path}")
        
        # Load model
        self.model_trainer.load_model(model_path)
        
        # Create test data loader
        test_loader, _ = self.model_trainer.create_data_loaders(
            test_data, test_data,
            batch_size=self.config['model_config']['batch_size'],
            max_length=self.config['model_config']['max_length']
        )
        
        # Evaluate
        accuracy, loss = self.model_trainer._evaluate_classification_model(test_loader)
        
        # Test individual predictions
        sample_predictions = []
        for i, row in test_data.head(10).iterrows():
            prediction = self.model_trainer.predict(row['question_text'])
            sample_predictions.append({
                'question': row['question_text'],
                'predicted_subject': prediction['subject'],
                'actual_subject': row.get('subject', 'Unknown'),
                'confidence': prediction['confidence']
            })
        
        evaluation_results = {
            'test_accuracy': accuracy,
            'test_loss': loss,
            'sample_predictions': sample_predictions,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model evaluation complete - Accuracy: {accuracy:.4f}")
        return evaluation_results
        
    def generate_training_report(self, results: Dict) -> str:
        """Generate a comprehensive training report."""
        report = f"""
# Government Exam AI Model Training Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- Base Model: {results['model_config']['base_model']}
- Epochs: {results['model_config']['epochs']}
- Batch Size: {results['model_config']['batch_size']}
- Learning Rate: {results['model_config']['learning_rate']}
- Max Sequence Length: {results['model_config']['max_length']}

## Dataset Statistics
- Total Questions: {results['dataset_stats']['total_questions']}
- Data Sources: {results['dataset_stats']['sources']}
- Subject Distribution: {results['dataset_stats']['subjects']}
- Difficulty Distribution: {results['dataset_stats']['difficulty_levels']}

## Training Results
- Final Test Accuracy: {results['test_accuracy']:.4f}
- Final Test Loss: {results['test_loss']:.4f}

## Training History
"""
        
        for epoch_data in results['training_history']:
            report += f"""
Epoch {epoch_data['epoch']}:
- Train Loss: {epoch_data['train_loss']:.4f}
- Train Accuracy: {epoch_data['train_accuracy']:.4f}
- Validation Loss: {epoch_data['val_loss']:.4f}
- Validation Accuracy: {epoch_data['val_accuracy']:.4f}
"""
        
        if results.get('fine_tuned'):
            report += f"""
## Fine-tuning Information
- Base Model: {results['base_model_path']}
- Fine-tuning completed successfully
"""
        
        report += f"""
## Next Steps
1. Deploy the trained model to production
2. Collect more training data to improve accuracy
3. Consider domain-specific fine-tuning for specific exam types
4. Implement continuous learning pipeline

---
Generated by Government Exam AI Training Pipeline
"""
        
        return report
        
    def export_for_production(self, model_path: str, export_path: str) -> None:
        """Export model and configuration for production deployment."""
        logger.info(f"Exporting model to {export_path}")
        
        # Create export directory
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        import shutil
        shutil.copytree(model_path, export_path / 'model', dirs_exist_ok=True)
        
        # Create production configuration
        production_config = {
            'model_name': self.model_trainer.model_name,
            'exam_categories': self.model_trainer.exam_config,
            'device': 'cpu',  # Production typically uses CPU
            'max_length': self.config['model_config']['max_length'],
            'created_timestamp': datetime.now().isoformat()
        }
        
        with open(export_path / 'production_config.json', 'w') as f:
            json.dump(production_config, f, indent=2)
            
        # Create prediction script
        prediction_script = '''#!/usr/bin/env python3
"""
Production prediction script for Government Exam AI
"""

import json
import torch
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.model_trainer import ModelTrainer

def load_production_model(config_path):
    """Load model for production predictions."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    trainer = ModelTrainer(config['model_name'])
    trainer.load_model(str(Path(__file__).parent / 'model'))
    
    return trainer, config

def predict_question(text, trainer, config):
    """Make prediction on a single question."""
    return trainer.predict(text)

if __name__ == "__main__":
    # Load model and config
    config_path = Path(__file__).parent / 'production_config.json'
    trainer, config = load_production_model(str(config_path))
    
    # Example usage
    test_questions = [
        "What is the capital of India?",
        "Calculate the compound interest for Rs. 10,000 at 5% for 2 years."
    ]
    
    for question in test_questions:
        prediction = predict_question(question, trainer, config)
        print(f"Question: {question}")
        print(f"Prediction: {prediction}")
        print("-" * 50)
'''
        
        with open(export_path / 'predict.py', 'w') as f:
            f.write(prediction_script)
            
        # Create requirements file
        requirements = '''torch
transformers
scikit-learn
pandas
numpy
nltk
'''
        
        with open(export_path / 'requirements.txt', 'w') as f:
            f.write(requirements)
            
        logger.info(f"Model exported successfully to {export_path}")


def main():
    """Main training pipeline execution."""
    parser = argparse.ArgumentParser(description='Government Exam AI Model Training')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data-sources', nargs='+', default=[], help='Additional data source files')
    parser.add_argument('--mode', choices=['train', 'fine_tune', 'evaluate'], default='train',
                       help='Training mode')
    parser.add_argument('--model-path', type=str, help='Path to existing model for fine-tuning/evaluation')
    parser.add_argument('--export', type=str, help='Export path for production deployment')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GovernmentExamModelPipeline(args.config)
    
    try:
        if args.mode == 'train':
            # Collect and prepare data
            data = pipeline.collect_and_prepare_data(args.data_sources)
            
            # Train model
            results = pipeline.train_model(data)
            
            # Generate and save report
            report = pipeline.generate_training_report(results)
            with open('training_report.md', 'w') as f:
                f.write(report)
                
            logger.info("Training completed successfully!")
            
        elif args.mode == 'fine_tune':
            if not args.model_path:
                raise ValueError("Model path required for fine-tuning")
                
            data = pipeline.collect_and_prepare_data(args.data_sources)
            results = pipeline.fine_tune_existing_model(args.model_path, data)
            
            logger.info("Fine-tuning completed successfully!")
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("Model path required for evaluation")
                
            data = pipeline.collect_and_prepare_data(args.data_sources)
            _, _, test_df = pipeline.dataset_builder.get_training_ready_data()
            results = pipeline.evaluate_model(args.model_path, test_df)
            
            logger.info(f"Evaluation results: {results}")
            
        # Export for production if requested
        if args.export:
            if not args.model_path and args.mode == 'train':
                args.model_path = pipeline.config['model_config']['save_dir']
            elif not args.model_path:
                raise ValueError("Model path required for export")
                
            pipeline.export_for_production(args.model_path, args.export)
            logger.info("Model exported for production!")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()