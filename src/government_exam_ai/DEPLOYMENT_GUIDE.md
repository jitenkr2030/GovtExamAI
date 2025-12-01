# Government Exam AI System - Deployment Guide

## System Architecture Overview

This comprehensive AI/ML system covers 150+ government exams across 15 categories, providing:

- **Question Classification**: AI-powered subject/topic/difficulty classification
- **Answer Evaluation**: NLP-based evaluation for both objective and subjective answers
- **Mock Test Generation**: Adaptive test generation based on student performance
- **Performance Analytics**: Detailed analytics and predictive insights
- **Multi-Exam Support**: UPSC, SSC, Banking, Railways, Defence, State Government exams

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js (optional, for frontend development)
- 4GB+ RAM recommended
- Storage: 2GB+ for data and models

### Installation

1. **Clone/Copy the project**
```bash
# If using git
git clone <repository-url>
cd government_exam_ai

# Or copy the entire project directory
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLP models** (run these commands):
```bash
python -c "import nltk; nltk.download('all')"
python -m spacy download en_core_web_sm
```

4. **Set up directory structure**
```bash
# The system creates directories automatically, but ensure:
mkdir -p data/processed
mkdir -p tests/generated
mkdir -p analytics/reports
mkdir -p models
```

### Running the Application

#### Option 1: Backend API Only
```bash
cd api
python main.py
```
- API will be available at: http://localhost:8000
- Interactive documentation: http://localhost:8000/docs

#### Option 2: Full Stack (Backend + Frontend)
1. **Start backend**:
```bash
cd api
python main.py
```

2. **Open frontend**:
- Open `frontend/index.html` in a web browser
- Or serve with a local server:
```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000

# Using Node.js serve (if installed)
npx serve .
```

3. **Access the application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## API Endpoints

### Core Endpoints
- `GET /` - System overview and status
- `GET /exams` - Get all supported exams
- `POST /classify-question` - Classify question by subject/topic/difficulty
- `POST /evaluate-answer` - Evaluate student answers
- `POST /generate-test` - Generate adaptive mock tests
- `POST /generate-standard-test` - Generate standard tests
- `POST /student-analytics` - Get student performance analytics
- `GET /student-profile/{student_id}` - Get student profile
- `GET /question-bank/stats` - Question bank statistics

### Usage Examples

#### Classify a Question
```python
import requests

response = requests.post('http://localhost:8000/classify-question', json={
    "text": "What is the capital of France?",
    "options": ["London", "Berlin", "Paris", "Madrid"],
    "exam_code": "ssc_cgl"
})
result = response.json()
print(result['classification'])
```

#### Evaluate an Answer
```python
response = requests.post('http://localhost:8000/evaluate-answer', json={
    "question_id": "Q001",
    "student_answer": "Paris",
    "correct_answer": "C",
    "answer_type": "objective",
    "max_marks": 1.0
})
result = response.json()
print(result['evaluation'])
```

#### Generate Test
```python
response = requests.post('http://localhost:8000/generate-test', json={
    "student_id": "student_001",
    "exam_code": "ssc_cgl",
    "total_questions": 50,
    "duration_minutes": 60,
    "adaptive": True
})
result = response.json()
print(result['test'])
```

## Configuration

### Exam Configuration
Modify `config/exam_categories.py` to:
- Add new exams
- Update exam patterns
- Modify subject distributions
- Change difficulty classifications

### Data Sources
Configure data sources in `config/data_sources.json`:
- Add websites for data scraping
- Set up database connections
- Configure file sources

### Model Parameters
Adjust ML model parameters in respective modules:
- Question classifier: `ml_models/question_classifier.py`
- Answer evaluator: `evaluation/answer_evaluator.py`
- Test generator: `test_generation/mock_test_generator.py`

## Data Management

### Adding Questions
1. **Manual addition**: Add to question bank through API
2. **Bulk upload**: Use `/upload-data` endpoint with PDF/Excel files
3. **Database import**: Place files in `data/processed/` directory

### Student Data
- Profiles automatically created on first use
- Stored in `data/student_profiles/` directory
- Analytics reports in `analytics/reports/`

### Test Data
- Generated tests saved in `tests/generated/`
- Question bank statistics available via API

## Advanced Features

### Custom Exam Integration
To add a new exam:
1. Update exam configuration in `config/exam_categories.py`
2. Add question data with the new exam code
3. Update frontend dropdowns if needed

### Model Training
To train custom models:
1. Prepare training data in `data/processed/`
2. Run training scripts in respective modules
3. Save models to `models/` directory

### Analytics Customization
Customize analytics by:
- Modifying `analytics/performance_analytics.py`
- Adding new metrics in `PerformanceMetrics`
- Extending `AnalyticsReport` structure

## Deployment Options

### Local Development
- Use the quick start instructions above
- Ideal for development and testing

### Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment
- **AWS**: Deploy on EC2, ECS, or Lambda
- **Google Cloud**: Use App Engine or Cloud Run
- **Azure**: Deploy on App Service or Container Instances
- **Heroku**: Use Docker deployment

### Production Considerations
- Use proper database (PostgreSQL/MongoDB)
- Implement user authentication
- Add caching (Redis)
- Set up monitoring and logging
- Use load balancers for scalability
- Implement rate limiting

## Monitoring and Maintenance

### Health Checks
- Monitor API response times
- Check memory usage
- Track model performance
- Monitor data quality

### Regular Updates
- Update exam syllabi
- Refresh question banks
- Retrain models periodically
- Update dependencies

### Performance Optimization
- Cache frequently accessed data
- Optimize database queries
- Use CDN for frontend assets
- Implement async processing

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies installed
2. **Model Loading**: Check model files exist
3. **Database Connections**: Verify database configurations
4. **Memory Issues**: Increase system resources

### Logs and Debugging
- Check console output for error messages
- Monitor log files in application directory
- Use FastAPI interactive documentation for API testing
- Enable debug mode in development

## Support and Documentation

### Documentation
- API documentation available at `/docs` endpoint
- Code comments throughout the codebase
- Configuration files are well-documented

### Getting Help
- Check the troubleshooting section
- Review error logs
- Consult the interactive API documentation
- Examine sample code in each module

## Security Considerations

### Data Protection
- Student data is sensitive - implement proper encryption
- Use HTTPS in production
- Implement proper authentication/authorization
- Regular security updates

### API Security
- Rate limiting for API endpoints
- Input validation and sanitization
- CORS configuration for frontend
- Secure headers and sessions

---

## Project Summary

This Government Exam AI System provides a comprehensive solution for exam preparation with:

✅ **150+ Government Exams** across 15 categories  
✅ **AI-Powered Question Classification** using NLP  
✅ **Intelligent Answer Evaluation** for objective and subjective answers  
✅ **Adaptive Test Generation** based on student performance  
✅ **Comprehensive Analytics** with predictive insights  
✅ **Scalable Architecture** supporting multiple exam formats  
✅ **RESTful API** for integration with other systems  
✅ **Web Interface** for easy interaction  
✅ **Modular Design** for easy customization and extension  

The system is designed to handle the complete exam preparation lifecycle, from data ingestion to performance analytics, making it suitable for students, educators, and coaching institutes.