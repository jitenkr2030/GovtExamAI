"""
Government Exam AI - Custom Transformer Training Demo
Author: MiniMax Agent
Date: 2025-12-01

This demo script shows how to use the custom transformer training system
for Indian government exam question classification and answer evaluation.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import our custom modules
from ml_models.model_trainer import ModelTrainer
from data_ingestion.data_collection_pipeline import DatasetBuilder
from training_pipeline import GovernmentExamModelPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_exam_data():
    """Create comprehensive sample exam data for training."""
    logger.info("Creating sample exam data...")
    
    # Sample exam questions from various Indian government exams
    sample_data = []
    
    # UPSC Civil Services questions
    upsc_questions = [
        {
            'question_text': 'What are the fundamental rights enshrined in the Indian Constitution? Explain each in brief.',
            'subject': 'Polity',
            'topic': 'Fundamental Rights',
            'difficulty': 'Hard',
            'exam_type': 'UPSC',
            'year': 2023
        },
        {
            'question_text': 'Discuss the role of the Election Commission in ensuring free and fair elections in India.',
            'subject': 'Polity',
            'topic': 'Election Commission',
            'difficulty': 'Medium',
            'exam_type': 'UPSC',
            'year': 2023
        },
        {
            'question_text': 'Calculate the compound interest on Rs. 50,000 for 3 years at 8% per annum.',
            'subject': 'Mathematics',
            'topic': 'Simple and Compound Interest',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023
        }
    ]
    
    # Banking exam questions
    banking_questions = [
        {
            'question_text': 'What is the current repo rate set by RBI? Explain its impact on the economy.',
            'subject': 'Economics',
            'topic': 'Monetary Policy',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023
        },
        {
            'question_text': 'A person deposits Rs. 10,000 in a bank offering 6% simple interest. How much will he get after 5 years?',
            'subject': 'Mathematics',
            'topic': 'Simple Interest',
            'difficulty': 'Easy',
            'exam_type': 'Banking',
            'year': 2023
        },
        {
            'question_text': 'Explain the concept of Non-Performing Assets (NPAs) and their impact on banks.',
            'subject': 'Banking',
            'topic': 'NPAs',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023
        }
    ]
    
    # SSC questions
    ssc_questions = [
        {
            'question_text': 'Who was the first woman to fly in space?',
            'subject': 'Science',
            'topic': 'Space Technology',
            'difficulty': 'Easy',
            'exam_type': 'SSC',
            'year': 2023
        },
        {
            'question_text': 'The process of conversion of gaseous state directly to solid state is called:',
            'subject': 'Science',
            'topic': 'States of Matter',
            'difficulty': 'Medium',
            'exam_type': 'SSC',
            'year': 2023
        },
        {
            'question_text': 'In which year did India win its first Cricket World Cup?',
            'subject': 'Sports',
            'topic': 'Cricket',
            'difficulty': 'Easy',
            'exam_type': 'SSC',
            'year': 2023
        }
    ]
    
    # Railway exam questions
    railway_questions = [
        {
            'question_text': 'How many types of trains are there in Indian Railways?',
            'subject': 'General Knowledge',
            'topic': 'Indian Railways',
            'difficulty': 'Medium',
            'exam_type': 'Railway',
            'year': 2023
        },
        {
            'question_text': 'The headquarters of Northern Railway is located at:',
            'subject': 'General Knowledge',
            'topic': 'Indian Railways',
            'difficulty': 'Easy',
            'exam_type': 'Railway',
            'year': 2023
        }
    ]
    
    # Defence exam questions
    defence_questions = [
        {
            'question_text': 'Who is the current Chief of Army Staff of Indian Army?',
            'subject': 'Defence',
            'topic': 'Army Leadership',
            'difficulty': 'Medium',
            'exam_type': 'Defence',
            'year': 2023
        },
        {
            'question_text': 'What is the motto of Indian Navy?',
            'subject': 'Defence',
            'topic': 'Navy',
            'difficulty': 'Easy',
            'exam_type': 'Defence',
            'year': 2023
        }
    ]
    
    # Combine all questions
    all_questions = (upsc_questions + banking_questions + ssc_questions + 
                    railway_questions + defence_questions)
    
    # Create multiple variations for training
    augmented_data = []
    base_questions = all_questions
    
    for question in base_questions:
        # Add original question
        augmented_data.append(question)
        
        # Create variations for training
        for i in range(4):  # 4 variations per question
            variation = question.copy()
            variation['question_text'] = question['question_text'] + f" (Variation {i+1})"
            variation['source'] = 'augmented'
            augmented_data.append(variation)
    
    logger.info(f"Created {len(augmented_data)} sample questions")
    return augmented_data


def save_sample_data():
    """Save sample data to files for demonstration."""
    logger.info("Saving sample data files...")
    
    # Create data directory
    data_dir = Path('data/sample')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    sample_data = create_sample_exam_data()
    
    # Save as JSON
    with open(data_dir / 'sample_exam_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(data_dir / 'sample_exam_data.csv', index=False, encoding='utf-8')
    
    # Create a larger dataset by adding more variations
    large_dataset = []
    for base_question in sample_data:
        for _ in range(10):  # Create 10 variations of each question
            variation = base_question.copy()
            variation['question_text'] = f"Q: {variation['question_text']}"
            variation['year'] = np.random.choice([2020, 2021, 2022, 2023])
            large_dataset.append(variation)
    
    with open(data_dir / 'large_exam_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(large_dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample data saved to {data_dir}")
    return data_dir


def demo_data_collection():
    """Demonstrate data collection and preprocessing."""
    logger.info("=" * 60)
    logger.info("DEMO: Data Collection and Preprocessing")
    logger.info("=" * 60)
    
    # Save sample data
    data_dir = save_sample_data()
    
    # Initialize dataset builder
    builder = DatasetBuilder()
    
    # Add sample data sources
    builder.add_data_source(str(data_dir / 'sample_exam_data.json'), 'json')
    builder.add_data_source(str(data_dir / 'sample_exam_data.csv'), 'csv')
    
    # Process and enhance data
    processed_df = builder.process_and_enhance_data()
    
    logger.info(f"Processed {len(processed_df)} questions")
    logger.info("Sample processed data:")
    logger.info(processed_df.head()[['question_text', 'subject', 'topic', 'difficulty']])
    
    # Show dataset statistics
    stats = builder.dataset_stats
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total questions: {stats['total_questions']}")
    logger.info(f"  Subjects: {stats['subjects']}")
    logger.info(f"  Topics: {list(stats['topics'].keys())}")
    logger.info(f"  Difficulty levels: {stats['difficulty_levels']}")
    
    return processed_df


def demo_model_training():
    """Demonstrate custom transformer model training."""
    logger.info("=" * 60)
    logger.info("DEMO: Custom Transformer Model Training")
    logger.info("=" * 60)
    
    # Initialize model trainer
    trainer = ModelTrainer(model_name='microsoft/DialoGPT-medium')
    
    # Create sample training data (smaller dataset for demo)
    sample_questions = [
        {'question_text': 'What is the capital of India?', 'subject': 'Geography', 'topic': 'Indian Geography', 'difficulty': 'Easy'},
        {'question_text': 'Calculate 2+2=', 'subject': 'Mathematics', 'topic': 'Arithmetic', 'difficulty': 'Easy'},
        {'question_text': 'Who is the father of Indian Constitution?', 'subject': 'Polity', 'topic': 'Constitution', 'difficulty': 'Easy'},
        {'question_text': 'What is photosynthesis?', 'subject': 'Science', 'topic': 'Biology', 'difficulty': 'Medium'},
        {'question_text': 'Explain Newton\'s first law', 'subject': 'Science', 'topic': 'Physics', 'difficulty': 'Medium'},
        {'question_text': 'What is GDP?', 'subject': 'Economics', 'topic': 'Macroeconomics', 'difficulty': 'Medium'},
    ] * 20  # Multiply for training
    
    df = pd.DataFrame(sample_questions)
    
    logger.info(f"Training on {len(df)} questions")
    
    # Initialize model
    trainer.initialize_model(dropout=0.3)
    
    # Create data loaders (using same data for train/val for demo)
    train_loader, val_loader = trainer.create_data_loaders(df, batch_size=8, max_length=128)
    
    # Train model (reduced epochs for demo)
    logger.info("Starting model training...")
    training_history = trainer.train_classification_model(
        train_loader, val_loader, 
        epochs=2,  # Reduced for demo
        learning_rate=1e-4
    )
    
    # Test predictions
    logger.info("Testing model predictions...")
    test_questions = [
        "What is the currency of India?",
        "Calculate 5+7",
        "Who wrote the Indian Constitution?",
        "What is atomic number of hydrogen?"
    ]
    
    predictions = []
    for question in test_questions:
        prediction = trainer.predict(question)
        predictions.append({
            'question': question,
            'subject': prediction['subject'],
            'difficulty': prediction['difficulty'],
            'confidence': prediction['confidence']
        })
        logger.info(f"Q: {question}")
        logger.info(f"  Predicted Subject: {prediction['subject']}")
        logger.info(f"  Predicted Difficulty: {prediction['difficulty']}")
    
    # Save model
    save_dir = Path('models/demo_model')
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    
    logger.info(f"Model saved to {save_dir}")
    
    return training_history, predictions


def demo_full_pipeline():
    """Demonstrate the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("DEMO: Complete Training Pipeline")
    logger.info("=" * 60)
    
    # Create training configuration
    config = {
        'model_config': {
            'base_model': 'microsoft/DialoGPT-medium',
            'epochs': 2,  # Reduced for demo
            'batch_size': 8,
            'learning_rate': 1e-4,
            'max_length': 256,
            'dropout': 0.3,
            'save_dir': 'models/pipeline_demo_model'
        },
        'dataset_config': {
            'min_text_length': 10,
            'augmentation_factor': 2,
            'deduplicate': True,
            'test_size': 0.3,
            'validation_size': 0.2
        },
        'training_settings': {
            'use_augmentation': True,
            'save_intermediate': True,
            'early_stopping': False
        }
    }
    
    # Save config
    config_path = Path('config/training_config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize pipeline
    pipeline = GovernmentExamModelPipeline(str(config_path))
    
    # Collect and prepare data
    processed_data = demo_data_collection()
    
    # Train model
    results = pipeline.train_model(processed_data)
    
    # Generate training report
    report = pipeline.generate_training_report(results)
    
    # Save report
    report_path = Path('reports/training_report_demo.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Training report saved to {report_path}")
    
    return results


def demo_production_export():
    """Demonstrate model export for production."""
    logger.info("=" * 60)
    logger.info("DEMO: Production Export")
    logger.info("=" * 60)
    
    # Check if we have a trained model
    model_path = Path('models/demo_model')
    if not model_path.exists():
        logger.warning("No trained model found. Training demo model first...")
        demo_model_training()
    
    # Export for production
    export_path = Path('production/government_exam_ai')
    export_path.mkdir(parents=True, exist_ok=True)
    
    pipeline = GovernmentExamModelPipeline()
    pipeline.export_for_production(str(model_path), str(export_path))
    
    logger.info(f"Model exported for production to {export_path}")
    
    # Test production prediction
    logger.info("Testing production prediction...")
    prediction_script_path = export_path / 'predict.py'
    
    if prediction_script_path.exists():
        logger.info("Production prediction script created successfully")
        logger.info(f"Script location: {prediction_script_path}")
        
        # Example prediction code
        test_prediction = '''
# Example usage of production model:
from predict import load_production_model, predict_question

# Load model
trainer, config = load_production_model('production_config.json')

# Make predictions
question = "What is the capital of France?"
prediction = predict_question(question, trainer, config)
print(f"Prediction: {prediction}")
'''
        
        logger.info("Example production usage:")
        logger.info(test_prediction)


def main():
    """Run all demonstration functions."""
    logger.info("Government Exam AI - Custom Transformer Training Demo")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Data Collection
        processed_data = demo_data_collection()
        
        # Demo 2: Model Training
        training_history, predictions = demo_model_training()
        
        # Demo 3: Full Pipeline
        pipeline_results = demo_full_pipeline()
        
        # Demo 4: Production Export
        demo_production_export()
        
        logger.info("=" * 80)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Summary
        logger.info("Demo Summary:")
        logger.info(f"✓ Created and processed {len(processed_data)} sample questions")
        logger.info(f"✓ Trained custom transformer model")
        logger.info(f"✓ Achieved training accuracy: {training_history[-1]['train_accuracy']:.4f}")
        logger.info(f"✓ Generated {len(predictions)} test predictions")
        logger.info(f"✓ Exported model for production deployment")
        
        logger.info("\nNext Steps:")
        logger.info("1. Collect real exam data from PDF files")
        logger.info("2. Run full training with larger dataset")
        logger.info("3. Deploy to production environment")
        logger.info("4. Implement continuous learning pipeline")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()