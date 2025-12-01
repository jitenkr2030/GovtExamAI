"""
Simple Demo of Government Exam AI - Custom Transformer Training
Author: MiniMax Agent
Date: 2025-12-01

This demo shows the complete data collection and preprocessing pipeline
without requiring heavy transformer model downloads.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import our custom modules
from data_ingestion.data_collection_pipeline import DatasetBuilder, DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comprehensive_sample_data():
    """Create a comprehensive sample dataset for training."""
    logger.info("Creating comprehensive sample exam data...")
    
    # Sample questions from various Indian government exams
    sample_questions = [
        # UPSC Questions
        {
            'question_text': 'What are the fundamental rights enshrined in the Indian Constitution? Explain each in brief.',
            'subject': 'Polity',
            'topic': 'Fundamental Rights',
            'difficulty': 'Hard',
            'exam_type': 'UPSC',
            'year': 2023,
            'source': 'upsc_cse_mains'
        },
        {
            'question_text': 'Discuss the role of the Election Commission in ensuring free and fair elections in India.',
            'subject': 'Polity',
            'topic': 'Election Commission',
            'difficulty': 'Medium',
            'exam_type': 'UPSC',
            'year': 2023,
            'source': 'upsc_cse_mains'
        },
        {
            'question_text': 'Explain the process of photosynthesis in plants.',
            'subject': 'Science',
            'topic': 'Biology',
            'difficulty': 'Medium',
            'exam_type': 'UPSC',
            'year': 2023,
            'source': 'upsc_cds'
        },
        
        # Banking Questions
        {
            'question_text': 'What is the current repo rate set by RBI? Explain its impact on the economy.',
            'subject': 'Economics',
            'topic': 'Monetary Policy',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023,
            'source': 'rbi_gradeb'
        },
        {
            'question_text': 'A person deposits Rs. 10,000 in a bank offering 6% simple interest. How much will he get after 5 years?',
            'subject': 'Mathematics',
            'topic': 'Simple Interest',
            'difficulty': 'Easy',
            'exam_type': 'Banking',
            'year': 2023,
            'source': 'ibps_po_prelims'
        },
        {
            'question_text': 'Explain the concept of Non-Performing Assets (NPAs) and their impact on banks.',
            'subject': 'Banking',
            'topic': 'NPAs',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023,
            'source': 'sbi_po_mains'
        },
        
        # SSC Questions
        {
            'question_text': 'Who was the first woman to fly in space?',
            'subject': 'Science',
            'topic': 'Space Technology',
            'difficulty': 'Easy',
            'exam_type': 'SSC',
            'year': 2023,
            'source': 'ssc_cgl_tier1'
        },
        {
            'question_text': 'The process of conversion of gaseous state directly to solid state is called:',
            'subject': 'Science',
            'topic': 'States of Matter',
            'difficulty': 'Medium',
            'exam_type': 'SSC',
            'year': 2023,
            'source': 'ssc_cgl_tier1'
        },
        {
            'question_text': 'In which year did India win its first Cricket World Cup?',
            'subject': 'Sports',
            'topic': 'Cricket',
            'difficulty': 'Easy',
            'exam_type': 'SSC',
            'year': 2023,
            'source': 'ssc_chsl_tier1'
        },
        
        # Railway Questions
        {
            'question_text': 'How many types of trains are there in Indian Railways?',
            'subject': 'General Knowledge',
            'topic': 'Indian Railways',
            'difficulty': 'Medium',
            'exam_type': 'Railway',
            'year': 2023,
            'source': 'rrb_ntpc'
        },
        {
            'question_text': 'The headquarters of Northern Railway is located at:',
            'subject': 'General Knowledge',
            'topic': 'Indian Railways',
            'difficulty': 'Easy',
            'exam_type': 'Railway',
            'year': 2023,
            'source': 'rrb_ntpc'
        },
        
        # Defence Questions
        {
            'question_text': 'Who is the current Chief of Army Staff of Indian Army?',
            'subject': 'Defence',
            'topic': 'Army Leadership',
            'difficulty': 'Medium',
            'exam_type': 'Defence',
            'year': 2023,
            'source': 'nda'
        },
        {
            'question_text': 'What is the motto of Indian Navy?',
            'subject': 'Defence',
            'topic': 'Navy',
            'difficulty': 'Easy',
            'exam_type': 'Defence',
            'year': 2023,
            'source': 'cds'
        },
        
        # More varied questions for better training
        {
            'question_text': 'Calculate the compound interest on Rs. 50,000 for 3 years at 8% per annum.',
            'subject': 'Mathematics',
            'topic': 'Compound Interest',
            'difficulty': 'Medium',
            'exam_type': 'Banking',
            'year': 2023,
            'source': 'ibps_po_prelims'
        },
        {
            'question_text': 'What is the capital of France?',
            'subject': 'Geography',
            'topic': 'World Geography',
            'difficulty': 'Easy',
            'exam_type': 'SSC',
            'year': 2023,
            'source': 'ssc_cgl_tier1'
        },
        {
            'question_text': 'Explain Newton\'s first law of motion.',
            'subject': 'Science',
            'topic': 'Physics',
            'difficulty': 'Medium',
            'exam_type': 'UPSC',
            'year': 2023,
            'source': 'upsc_cds'
        },
        {
            'question_text': 'What is the GDP of India in 2023?',
            'subject': 'Economics',
            'topic': 'Macroeconomics',
            'difficulty': 'Hard',
            'exam_type': 'Banking',
            'year': 2023,
            'source': 'rbi_gradeb'
        },
        {
            'question_text': 'Which article of the Indian Constitution deals with the right to equality?',
            'subject': 'Polity',
            'topic': 'Fundamental Rights',
            'difficulty': 'Medium',
            'exam_type': 'UPSC',
            'year': 2023,
            'source': 'upsc_cse_prelims'
        }
    ]
    
    # Create variations for training data augmentation
    augmented_questions = []
    for question in sample_questions:
        # Add original question
        augmented_questions.append(question)
        
        # Create 2-3 variations per question
        for i in range(2):
            variation = question.copy()
            variation['question_text'] = f"{question['question_text']} (Variant {i+1})"
            variation['source'] = 'augmented'
            augmented_questions.append(variation)
    
    logger.info(f"Created {len(augmented_questions)} sample questions with variations")
    return augmented_questions


def demo_data_collection_pipeline():
    """Demonstrate the complete data collection and preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("DEMO: Data Collection and Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Create sample data
    sample_data = create_comprehensive_sample_data()
    
    # Initialize dataset builder
    builder = DatasetBuilder()
    
    # Add sample data
    builder.raw_data = sample_data
    
    # Process and enhance data
    logger.info("Processing and enhancing collected data...")
    processed_df = builder.process_and_enhance_data(
        min_text_length=10,  # Lower threshold for demo
        deduplicate=True
    )
    
    # Display results
    logger.info(f"Processed {len(processed_df)} questions")
    
    # Show sample processed data
    logger.info("Sample processed data:")
    display_cols = ['question_text', 'subject', 'topic', 'difficulty', 'text_length', 'word_count']
    if all(col in processed_df.columns for col in display_cols):
        sample_display = processed_df[display_cols].head(5)
        for idx, row in sample_display.iterrows():
            logger.info(f"Q: {row['question_text'][:60]}...")
            logger.info(f"  Subject: {row['subject']}, Topic: {row['topic']}, Difficulty: {row['difficulty']}")
            logger.info(f"  Length: {row['text_length']} chars, Words: {row['word_count']}")
            logger.info("")
    
    # Show dataset statistics
    stats = builder.dataset_stats
    logger.info("Dataset Statistics:")
    logger.info(f"  Total questions: {stats['total_questions']}")
    logger.info(f"  Sources: {stats['sources']}")
    logger.info(f"  Subjects: {stats['subjects']}")
    logger.info(f"  Topics: {list(stats['topics'].keys())}")
    logger.info(f"  Difficulty levels: {stats['difficulty_levels']}")
    logger.info(f"  Text statistics: {stats['text_statistics']}")
    
    # Demonstrate data augmentation
    logger.info("\nDemonstrating data augmentation...")
    original_count = len(processed_df)
    augmented_df = builder.augment_dataset(augmentation_factor=2)
    augmented_count = len(augmented_df)
    
    logger.info(f"Original dataset: {original_count} questions")
    logger.info(f"After augmentation: {augmented_count} questions")
    logger.info(f"Augmentation factor: {augmented_count/original_count:.2f}x")
    
    # Save processed dataset
    save_path = Path('data/processed_exam_dataset')
    save_path.mkdir(parents=True, exist_ok=True)
    builder.save_dataset(str(save_path))
    
    logger.info(f"Dataset saved to {save_path}")
    
    return processed_df, builder


def demo_text_preprocessing():
    """Demonstrate advanced text preprocessing capabilities."""
    logger.info("=" * 60)
    logger.info("DEMO: Advanced Text Preprocessing")
    logger.info("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(use_spacy=False)  # Use basic preprocessing for demo
    
    # Test questions with various content types
    test_questions = [
        "What is the capital of India?",
        "Calculate the compound interest on Rs. 10,000 at 5% for 2 years.",
        "Explain the process of photosynthesis in plants with reference to light and dark reactions.",
        "The battle of Plassey was fought in which year? Explain its significance in Indian history.",
        "What are the fundamental rights guaranteed under Article 19 of the Indian Constitution?",
        "A train travels 240 km in 4 hours. What is its average speed?",
        "Who was the first woman to win the Nobel Prize?",
        "Define the term 'Inflation' and explain its impact on the economy."
    ]
    
    logger.info("Processing test questions...")
    processed_results = []
    
    for question in test_questions:
        logger.info(f"Original: {question}")
        
        # Clean text
        cleaned_text = preprocessor.clean_text(question)
        logger.info(f"Cleaned: {cleaned_text}")
        
        # Extract features
        features = preprocessor.extract_key_features(question)
        logger.info(f"Features: Length={features['text_length']}, Words={features['word_count']}, Sentences={features['sentence_count']}")
        logger.info(f"  Markers: {features['subject_markers']}")
        logger.info(f"  Math symbols: {features['has_math_symbols']}, Numbers: {features['has_numbers']}, Years: {features['has_years']}")
        logger.info(f"  Complexity: {features['difficulty_indicators']['complexity_score']:.3f}")
        logger.info("")
        
        processed_results.append({
            'question': question,
            'cleaned': cleaned_text,
            'features': features
        })
    
    return processed_results


def demo_data_source_integration():
    """Demonstrate integration with different data sources."""
    logger.info("=" * 60)
    logger.info("DEMO: Multi-Source Data Integration")
    logger.info("=" * 60)
    
    # Initialize dataset builder
    builder = DatasetBuilder()
    
    # Create sample files for different source types
    data_dir = Path('demo_data_sources')
    data_dir.mkdir(exist_ok=True)
    
    # JSON format
    json_data = [
        {
            'question_text': 'What is thecurrency of Japan?',
            'subject': 'Geography',
            'topic': 'Currencies',
            'difficulty': 'Easy',
            'exam_type': 'Banking'
        },
        {
            'question_text': 'Calculate the area of a circle with radius 7 cm.',
            'subject': 'Mathematics',
            'topic': 'Geometry',
            'difficulty': 'Medium',
            'exam_type': 'SSC'
        }
    ]
    
    with open(data_dir / 'sample_questions.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # CSV format
    csv_data = pd.DataFrame([
        {
            'question_text': 'Who wrote the Indian National Anthem?',
            'subject': 'General Knowledge',
            'topic': 'Indian Culture',
            'difficulty': 'Easy'
        },
        {
            'question_text': "Define the term 'Momentum' in physics.",
            'subject': 'Science',
            'topic': 'Physics',
            'difficulty': 'Medium'
        }
    ])
    
    csv_data.to_csv(data_dir / 'sample_questions.csv', index=False)
    
    # Text format
    with open(data_dir / 'sample_questions.txt', 'w') as f:
        f.write("What is the largest planet in our solar system?\n\n")
        f.write("Explain the concept of demand and supply in economics.\n\n")
        f.write("Who was the first Prime Minister of India?\n\n")
        f.write("Calculate the simple interest on Rs. 5000 at 4% for 3 years.")
    
    logger.info("Created sample data files:")
    logger.info(f"  JSON: {data_dir / 'sample_questions.json'}")
    logger.info(f"  CSV: {data_dir / 'sample_questions.csv'}")
    logger.info(f"  Text: {data_dir / 'sample_questions.txt'}")
    
    # Simulate adding these sources (without actually processing)
    logger.info("\nSimulating data source integration...")
    for source_file in data_dir.glob('*'):
        source_type = source_file.suffix[1:]  # Remove the dot
        logger.info(f"Would add {source_file} as type: {source_type}")
    
    return data_dir


def create_training_config():
    """Create a sample training configuration."""
    config = {
        'model_config': {
            'base_model': 'microsoft/DialoGPT-medium',
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'max_length': 512,
            'dropout': 0.3,
            'save_dir': 'models/government_exam_transformer'
        },
        'dataset_config': {
            'min_text_length': 20,
            'augmentation_factor': 3,
            'deduplicate': True,
            'test_size': 0.2,
            'validation_size': 0.1
        },
        'training_settings': {
            'use_augmentation': True,
            'save_intermediate': True,
            'early_stopping': True,
            'validation_metric': 'accuracy'
        },
        'exam_config': {
            'subjects': [
                'Mathematics', 'Science', 'History', 'Geography', 'Polity', 
                'Economics', 'Banking', 'General Knowledge', 'English', 
                'Reasoning', 'Current Affairs', 'Defence'
            ],
            'topics': [
                'Arithmetic', 'Algebra', 'Geometry', 'Physics', 'Chemistry', 
                'Biology', 'Ancient History', 'Modern History', 'Indian Geography',
                'World Geography', 'Indian Polity', 'Economics', 'Banking Operations'
            ],
            'difficulty_levels': ['Easy', 'Medium', 'Hard']
        }
    }
    
    config_path = Path('config/demo_training_config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training configuration saved to {config_path}")
    return config, config_path


def main():
    """Run the complete data pipeline demonstration."""
    logger.info("Government Exam AI - Data Collection & Preprocessing Demo")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Data Collection Pipeline
        processed_data, builder = demo_data_collection_pipeline()
        
        # Demo 2: Text Preprocessing
        text_results = demo_text_preprocessing()
        
        # Demo 3: Multi-source Integration
        demo_sources = demo_data_source_integration()
        
        # Demo 4: Training Configuration
        training_config, config_path = create_training_config()
        
        # Summary
        logger.info("=" * 80)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("Demo Summary:")
        logger.info(f"✓ Created and processed {len(processed_data)} sample questions")
        logger.info(f"✓ Processed {len(text_results)} test questions with advanced features")
        logger.info(f"✓ Created {len(list(demo_sources.glob('*')))} sample data source files")
        logger.info(f"✓ Generated training configuration: {config_path}")
        
        logger.info("\nNext Steps for Full Training:")
        logger.info("1. Collect real exam data from PDF files")
        logger.info("2. Install PyTorch and transformers for deep learning")
        logger.info("3. Run: python training_pipeline.py --mode train")
        logger.info("4. Fine-tune with: python training_pipeline.py --mode fine_tune")
        logger.info("5. Deploy with: python training_pipeline.py --export production_model")
        
        logger.info("\nFiles Created:")
        logger.info(f"- Processed dataset: data/processed_exam_dataset/")
        logger.info(f"- Training config: {config_path}")
        logger.info(f"- Sample data sources: {demo_sources}")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All demos completed successfully!")
        print("Your government exam AI training system is ready!")
    else:
        print("\n❌ Demo encountered errors. Check the logs above.")