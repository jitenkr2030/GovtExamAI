# Government Exam AI - Production Deployment

## 🚀 Quick Start

### Local Deployment

```bash
# Clone the repository
git clone <repository-url>
cd production_deployment

# Deploy locally
./deploy_local.sh

# Test the API
curl http://localhost:8000/health
```

### Cloud Deployment

```bash
# Deploy to AWS
./deploy_cloud.sh aws prod

# Deploy to Google Cloud Platform  
./deploy_cloud.sh gcp prod

# Deploy to Microsoft Azure
./deploy_cloud.sh azure prod
```

## 📋 System Architecture

### Components
- **FastAPI Service**: Core API server for question analysis
- **Model Inference**: Custom transformer for exam question classification
- **Load Balancer**: Nginx for request routing and SSL termination
- **Monitoring**: Prometheus + Grafana for metrics collection
- **Logging**: Centralized logging with ELK stack

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict/subject` | POST | Predict question subject |
| `/predict/difficulty` | POST | Predict difficulty level |
| `/classify/topic` | POST | Classify question topic |
| `/analyze` | POST | Full question analysis |
| `/metrics` | GET | API performance metrics |

### Request Example

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the capital of France?",
       "exam_type": "SSC CGL"
     }'
```

### Response Example

```json
{
  "question": "What is the capital of France?",
  "subject": {
    "predicted_subject": "General Knowledge",
    "confidence": 0.92,
    "alternatives": []
  },
  "topic": {
    "predicted_topic": "Geography", 
    "confidence": 0.88,
    "alternatives": []
  },
  "difficulty": {
    "predicted_difficulty": "Easy",
    "confidence": 0.95,
    "reasoning": "Simple concepts, direct application of basic facts"
  },
  "processing_time_ms": 125.6,
  "model_version": "1.0.0",
  "timestamp": "2025-12-01T15:01:37Z"
}
```

## ☁️ Cloud Deployment Options

### AWS (Recommended for scale)
- **Compute**: ECS Fargate or Lambda
- **Storage**: S3 for model artifacts
- **Database**: RDS for metadata
- **Monitoring**: CloudWatch
- **Cost**: $50-200/month

### Google Cloud Platform
- **Compute**: Cloud Run (serverless)
- **Storage**: Cloud Storage
- **Database**: Cloud SQL
- **Monitoring**: Cloud Monitoring
- **Cost**: $40-150/month

### Microsoft Azure  
- **Compute**: Container Instances
- **Storage**: Blob Storage
- **Database**: Azure SQL
- **Monitoring**: Application Insights
- **Cost**: $45-180/month

## 📊 Monitoring & Metrics

### Key Performance Indicators
- **Response Time**: < 200ms (P95)
- **Throughput**: 1000+ requests/minute
- **Availability**: 99.9% uptime
- **Model Accuracy**: > 80% for subject classification

### Monitoring Dashboards
1. **API Performance**: Response times, throughput, error rates
2. **Model Performance**: Accuracy metrics, inference times
3. **Infrastructure**: CPU, memory, disk usage
4. **Business Metrics**: Usage patterns, user engagement

## 🔒 Security Considerations

### API Security
- Rate limiting (1000 requests/minute)
- API key authentication
- Request validation
- CORS configuration
- SSL/TLS encryption

### Model Security
- Input sanitization
- Output filtering
- Model versioning
- Audit logging

## 🚀 Scaling & Performance

### Horizontal Scaling
- Auto-scaling based on CPU/memory usage
- Load balancing across multiple instances
- Stateless API design

### Performance Optimization
- Model caching
- Batch processing for multiple requests
- CDN for static content
- Database connection pooling

## 📝 Maintenance & Updates

### Regular Tasks
- Model retraining (monthly)
- Security updates (weekly)
- Performance monitoring (daily)
- Log rotation (weekly)

### Deployment Pipeline
1. Code changes → GitHub
2. Automated testing
3. Staging deployment
4. Production deployment
5. Smoke testing

## 🆘 Troubleshooting

### Common Issues

**High Response Times**
- Check model loading
- Monitor memory usage
- Verify database connections

**High Error Rates**  
- Check input validation
- Monitor model accuracy
- Review error logs

**Low Throughput**
- Scale horizontally
- Optimize model inference
- Check resource limits

### Log Locations
- Application logs: `/app/logs/gov_exam_ai.log`
- Nginx logs: `/var/log/nginx/`
- System logs: `journalctl -u gov-exam-ai`

## 📞 Support

For technical support or questions:
- Documentation: `/docs` endpoint
- Health check: `/health` endpoint  
- Metrics: `/metrics` endpoint
- Logs: Check application logs

## 📄 License

This project is licensed under the MIT License.
