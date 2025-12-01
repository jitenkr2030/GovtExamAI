# Government Exam AI - Custom Transformer Training System

A comprehensive AI system for training custom transformer models specifically designed for Indian government exam question classification and answer evaluation.

## 🚀 Features

- **Custom Transformer Models**: Train your own transformer models on government exam data
- **Multi-source Data Collection**: Process PDF exam papers, JSON files, CSV data, and text files
- **Advanced Preprocessing**: Smart text cleaning, feature extraction, and data augmentation
- **Multi-task Learning**: Classify subjects, topics, difficulty levels, and score answers
- **Production Ready**: Export trained models for deployment
- **Comprehensive Evaluation**: Detailed training reports and performance metrics

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for training)
- GPU optional (training works on CPU, faster with GPU)

## 🔧 Installation

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd government_exam_ai

# Install Python dependencies
pip install -r requirements.txt

# Download NLP models (required)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('all')"
```

### 2. Verify Installation

```bash
# Test the system
python demo_training.py
```

## 🎯 Quick Start

### Training with Sample Data

```python
from training_pipeline import GovernmentExamModelPipeline

# Initialize pipeline
pipeline = GovernmentExamModelPipeline()

# Collect and prepare data
data = pipeline.collect_and_prepare_data()

# Train model
results = pipeline.train_model(data)

# Generate training report
report = pipeline.generate_training_report(results)
print(report)
```

### Using Command Line

```bash
# Train with sample data
python training_pipeline.py --mode train

# Train with custom data sources
python training_pipeline.py --mode train --data-sources /path/to/exam_data.json

# Fine-tune existing model
python training_pipeline.py --mode fine_tune --model-path models/my_model

# Export for production
python training_pipeline.py --export production_model --model-path models/my_model
```

## 📊 Data Collection

### Supported Data Formats

1. **PDF Files**: Exam papers with structured questions
2. **JSON Files**: Pre-formatted exam data
3. **CSV Files**: Tabular exam data
4. **Text Files**: Raw text questions

### Data Structure

```json
{
  "question_text": "What is the capital of India?",
  "subject": "Geography",
  "topic": "Indian Geography",
  "difficulty": "Easy",
  "answer_text": "New Delhi",
  "exam_type": "UPSC",
  "year": 2023
}
```

### Adding Data Sources

```python
from data_ingestion.data_collection_pipeline import DatasetBuilder

# Initialize builder
builder = DatasetBuilder()

# Add PDF exam papers
builder.add_data_source('exams/paper1.pdf', 'pdf')

# Add JSON data
builder.add_data_source('data/exam_questions.json', 'json')

# Add CSV data
builder.add_data_source('data/compiled_questions.csv', 'csv')

# Add text files
builder.add_data_source('questions/text_questions.txt', 'text')

# Process all data
processed_data = builder.process_and_enhance_data()
```

## 🤖 Model Training

### Custom Transformer Architecture

The system uses a custom transformer architecture with:
- **Base Model**: Microsoft DialoGPT-medium (configurable)
- **Multi-task Heads**: Subject, Topic, Difficulty classification
- **Answer Scoring**: Regression head for answer evaluation
- **Dropout**: 0.3 for regularization

### Training Configuration

```python
config = {
    'model_config': {
        'base_model': 'microsoft/DialoGPT-medium',
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'max_length': 512,
        'dropout': 0.3
    },
    'dataset_config': {
        'min_text_length': 20,
        'augmentation_factor': 3,
        'test_size': 0.2,
        'validation_size': 0.1
    }
}
```

### Training Process

1. **Data Collection**: Gather exam data from various sources
2. **Preprocessing**: Clean, enhance, and augment data
3. **Model Initialization**: Load base transformer with custom heads
4. **Training**: Multi-task learning with validation
5. **Evaluation**: Test set evaluation and metrics
6. **Export**: Production-ready model export

## 🎯 Advanced Features

### Data Augmentation

```python
# Automatic data augmentation
augmented_data = builder.augment_dataset(augmentation_factor=3)

# Manual augmentation techniques
from data_ingestion.data_collection_pipeline import DataAugmentor

augmentor = DataAugmentor(preprocessor)
enhanced_questions = augmentor.augment_questions(questions, factor=5)
```

### Model Fine-tuning

```python
# Fine-tune existing model on new data
pipeline = GovernmentExamModelPipeline()
results = pipeline.fine_tune_existing_model(
    model_path='models/existing_model',
    data=new_exam_data
)
```

### Multi-source Training

```python
# Train on multiple data sources
sources = [
    'pdfs/2023_exams/*.pdf',
    'data/banking_questions.json',
    'data/sample_exams.csv'
]

results = pipeline.train_model(
    sources=sources,
    augmentation_factor=2
)
```

## 📈 Performance Optimization

### GPU Training

```python
# Enable GPU acceleration
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")
```

### Memory Optimization

```python
# Reduce batch size for memory-constrained environments
config['model_config']['batch_size'] = 4
config['model_config']['max_length'] = 256
```

### Dataset Optimization

```python
# Use sampling for large datasets
def sample_large_dataset(df, sample_size=10000):
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df
```

## 🔍 Model Evaluation

### Performance Metrics

```python
# Training results include
{
    'test_accuracy': 0.89,
    'test_loss': 0.25,
    'training_history': [...],
    'subject_accuracy': 0.92,
    'topic_accuracy': 0.87,
    'difficulty_accuracy': 0.85
}
```

### Evaluation Functions

```python
# Test model predictions
test_questions = [
    "What is photosynthesis?",
    "Calculate compound interest",
    "Explain Newton's laws"
]

for question in test_questions:
    prediction = trainer.predict(question)
    print(f"Q: {question}")
    print(f"Subject: {prediction['subject']}")
    print(f"Topic: {prediction['topic']}")
    print(f"Difficulty: {prediction['difficulty']}")
    print("-" * 50)
```

## 🚀 Production Deployment

### Export for Production

```python
# Export trained model
pipeline.export_for_production(
    model_path='models/trained_model',
    export_path='production/government_exam_ai'
)
```

### Production Usage

```python
from predict import load_production_model, predict_question

# Load production model
trainer, config = load_production_model('production_config.json')

# Make predictions
question = "What is the largest planet in our solar system?"
prediction = predict_question(question, trainer, config)
print(prediction)
```

### API Integration

```python
# Add to existing FastAPI app
from fastapi import FastAPI
from predict import load_production_model

app = FastAPI()
trainer, config = load_production_model('production_config.json')

@app.post("/predict")
async def predict_exam_question(question: dict):
    text = question.get('text', '')
    prediction = trainer.predict(text)
    return prediction
```

## 📊 Training Reports

The system generates comprehensive training reports:

```markdown
# Training Report

## Model Configuration
- Base Model: microsoft/DialoGPT-medium
- Epochs: 5
- Batch Size: 16
- Learning Rate: 1e-4

## Dataset Statistics
- Total Questions: 5000
- Subjects: 8 categories
- Difficulty Distribution: Easy (40%), Medium (45%), Hard (15%)

## Training Results
- Final Test Accuracy: 0.89
- Subject Classification: 92%
- Topic Classification: 87%
- Difficulty Prediction: 85%
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or max_length
2. **Slow Training**: Use GPU if available
3. **Poor Accuracy**: Increase dataset size or adjust augmentation
4. **Import Errors**: Install all requirements and download NLP models

### Performance Tips

1. **Data Quality**: Clean and validate input data
2. **Balanced Dataset**: Ensure representation across subjects/difficulty levels
3. **Regular Validation**: Monitor training progress
4. **Incremental Training**: Start with small datasets and scale up

## 📚 API Reference

### ModelTrainer

```python
class ModelTrainer:
    def __init__(self, model_name='microsoft/DialoGPT-medium')
    def initialize_model(self, dropout=0.3)
    def load_training_data(self, data_path)
    def preprocess_data(self, df)
    def train_classification_model(self, train_loader, val_loader)
    def predict(self, text, return_all_predictions=False)
    def save_model(self, save_dir)
    def load_model(self, save_dir)
```

### DatasetBuilder

```python
class DatasetBuilder:
    def add_data_source(self, source_path, source_type)
    def process_and_enhance_data(self, min_text_length=20)
    def augment_dataset(self, augmentation_factor=2)
    def save_dataset(self, save_path)
    def get_training_ready_data(self, test_size=0.2)
```

### GovernmentExamModelPipeline

```python
class GovernmentExamModelPipeline:
    def collect_and_prepare_data(self, sources=None)
    def train_model(self, data=None, save_model=True)
    def fine_tune_existing_model(self, model_path, data)
    def evaluate_model(self, model_path, test_data)
    def export_for_production(self, model_path, export_path)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

## 🔮 Future Enhancements

- [ ] Support for more transformer architectures
- [ ] Advanced data augmentation techniques
- [ ] Multi-language exam support
- [ ] Real-time model updates
- [ ] Integration with popular ML platforms

---

**Built with ❤️ for Indian Government Exam Aspirants**