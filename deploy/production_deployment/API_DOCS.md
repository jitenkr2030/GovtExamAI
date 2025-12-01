# Government Exam AI API Documentation

## Authentication

Currently no authentication required. In production, API keys should be implemented.

## Rate Limiting

- 1000 requests per minute per IP
- Burst limit: 100 requests per second

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2025-12-01T15:01:37Z"
}
```

## SDK Examples

### Python
```python
import requests

def analyze_question(question, exam_type="SSC CGL"):
    response = requests.post(
        "https://api.gov-exam-ai.com/analyze",
        json={
            "question": question,
            "exam_type": exam_type
        }
    )
    return response.json()

result = analyze_question("What is 2 + 2?")
print(result)
```

### JavaScript
```javascript
async function analyzeQuestion(question, examType = "SSC CGL") {
    const response = await fetch('https://api.gov-exam-ai.com/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            exam_type: examType
        })
    });
    return await response.json();
}

const result = await analyzeQuestion("What is 2 + 2?");
console.log(result);
```

### cURL
```bash
# Predict subject
curl -X POST "https://api.gov-exam-ai.com/predict/subject" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the capital of France?"}'

# Full analysis  
curl -X POST "https://api.gov-exam-ai.com/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Solve: x + 5 = 10",
       "options": ["x = 3", "x = 5", "x = 10", "x = 15"],
       "exam_type": "Mathematics"
     }'
```
