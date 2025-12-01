"""
FastAPI Backend for Government Exam AI System
Provides REST API endpoints for all system components
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import os
import json
import asyncio
from pathlib import Path
import logging
from datetime import datetime
import uvicorn

# Import our custom modules
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Ensure site-packages are in path for sklearn and other ML libs
import site
for sp in site.getsitepackages():
    if sp not in sys.path:
        sys.path.append(sp)

from config.exam_categories import get_exam_config, ExamConfig
from data_ingestion.data_pipeline import DataPipeline, ExamData
from ml_models.question_classifier import QuestionClassifier, Question, ClassificationResult
from evaluation.answer_evaluator import AnswerEvaluationEngine, Answer, EvaluationResult
from test_generation.mock_test_generator import (
    AdaptiveTestGenerator, QuestionBank, StudentProfiler, 
    TestSpecification, GeneratedTest
)
from analytics.performance_analytics import AnalyticsEngine, AnalyticsReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Government Exam AI System",
    description="AI-powered platform for 150+ government exam preparation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components initialization
question_bank = None
student_profiler = None
test_generator = None
analytics_engine = None
question_classifier = None
answer_evaluator = None

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    global question_bank, student_profiler, test_generator, analytics_engine, question_classifier, answer_evaluator
    
    try:
        logger.info("Initializing system components...")
        
        # Initialize question bank
        question_bank = QuestionBank()
        
        # Initialize student profiler
        student_profiler = StudentProfiler()
        
        # Initialize test generator
        test_generator = AdaptiveTestGenerator(question_bank, student_profiler)
        
        # Initialize analytics engine
        analytics_engine = AnalyticsEngine()
        
        # Initialize question classifier (would need training data)
        question_classifier = QuestionClassifier()
        
        # Initialize answer evaluator
        answer_evaluator = AnswerEvaluationEngine()
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")

# Pydantic models for API requests
class QuestionRequest(BaseModel):
    text: str
    options: List[str] = []
    correct_answer: Optional[str] = None
    exam_code: Optional[str] = None

class AnswerEvaluationRequest(BaseModel):
    question_id: str
    student_answer: str
    correct_answer: str
    answer_type: str = "objective"
    max_marks: float = 1.0
    subject: Optional[str] = None
    keywords: List[str] = []

class TestGenerationRequest(BaseModel):
    student_id: str
    exam_code: str
    total_questions: int = 50
    duration_minutes: int = 60
    adaptive: bool = True
    target_score: Optional[float] = None
    subject_distribution: Optional[Dict[str, float]] = None

class StudentPerformanceRequest(BaseModel):
    student_id: str
    days_back: int = 30

# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Government Exam AI System API",
        "version": "1.0.0",
        "status": "operational",
        "supported_exams": len(get_exam_config()),
        "endpoints": {
            "/exams": "Get all supported exams",
            "/classify-question": "Classify question by subject/topic/difficulty",
            "/evaluate-answer": "Evaluate student answer",
            "/generate-test": "Generate mock test",
            "/student-analytics": "Get student performance analytics",
            "/cohort-analytics": "Get cohort analytics"
        }
    }

@app.get("/exams")
async def get_supported_exams():
    """Get all supported government exams"""
    try:
        exam_config = get_exam_config()
        
        formatted_exams = {}
        for category, exams in exam_config.items():
            formatted_exams[category] = []
            for exam_code, exam_info in exams.items():
                formatted_exams[category].append({
                    "code": exam_code,
                    "name": exam_info["name"],
                    "full_name": exam_info["full_name"],
                    "stages": exam_info["stages"],
                    "subjects": exam_info["subjects"],
                    "difficulty": exam_info["difficulty"],
                    "total_marks": sum(exam_info["marks_pattern"].values())
                })
        
        return {"exams": formatted_exams, "total_categories": len(exam_config)}
        
    except Exception as e:
        logger.error(f"Error fetching exam configuration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch exam configuration")

@app.post("/classify-question")
async def classify_question(request: QuestionRequest):
    """Classify question by subject, topic, and difficulty"""
    try:
        if not question_classifier.is_trained:
            # For demo purposes, use rule-based classification
            classification = rule_based_classification(request.text, request.options)
        else:
            # Use trained ML model
            question = Question(
                text=request.text,
                options=request.options,
                correct_answer=request.correct_answer or "",
                exam_code=request.exam_code
            )
            result = question_classifier.predict(question)
            
            classification = {
                "predicted_subject": result.predicted_subject,
                "subject_confidence": result.subject_confidence,
                "predicted_topic": result.predicted_topic,
                "topic_confidence": result.topic_confidence,
                "predicted_difficulty": result.predicted_difficulty,
                "difficulty_confidence": result.difficulty_confidence,
                "subject_probabilities": result.subject_probabilities,
                "topic_probabilities": result.topic_probabilities,
                "difficulty_probabilities": result.difficulty_probabilities
            }
        
        return {"classification": classification, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error classifying question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/evaluate-answer")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Evaluate student answer"""
    try:
        answer = Answer(
            question_id=request.question_id,
            student_answer=request.student_answer,
            correct_answer=request.correct_answer,
            answer_type=request.answer_type,
            max_marks=request.max_marks,
            subject=request.subject,
            keywords=request.keywords
        )
        
        result = answer_evaluator.evaluate_answer(answer)
        
        return {
            "evaluation": {
                "question_id": result.question_id,
                "marks_awarded": result.marks_awarded,
                "max_marks": result.max_marks,
                "percentage": result.percentage,
                "feedback": result.feedback,
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
                "keyword_matches": result.keyword_matches,
                "missing_keywords": result.missing_keywords,
                "quality_score": result.quality_score,
                "detailed_scores": result.detailed_scores
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/generate-test")
async def generate_test(request: TestGenerationRequest):
    """Generate mock test for student"""
    try:
        # Get exam configuration
        exam_config = None
        all_exams = get_exam_config()
        for category_exams in all_exams.values():
            if request.exam_code in category_exams:
                exam_config = category_exams[request.exam_code]
                break
        
        if not exam_config:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        # Create subject distribution if not provided
        if not request.subject_distribution:
            subjects = exam_config["subjects"]
            equal_distribution = 100.0 / len(subjects)
            request.subject_distribution = {subject: equal_distribution for subject in subjects}
        
        # Create test specification
        specification = TestSpecification(
            exam_code=request.exam_code,
            total_questions=request.total_questions,
            total_marks=request.total_questions,  # Assuming 1 mark per question
            duration_minutes=request.duration_minutes,
            subject_distribution=request.subject_distribution,
            difficulty_distribution={'easy': 40, 'medium': 50, 'hard': 10},
            include_explanations=True,
            adaptive=request.adaptive,
            target_score=request.target_score
        )
        
        # Generate test
        generated_test = test_generator.generate_test(request.student_id, specification)
        
        # Convert to API response format
        test_response = {
            "test_id": generated_test.test_id,
            "student_id": generated_test.student_id,
            "specification": {
                "exam_code": generated_test.specification.exam_code,
                "total_questions": generated_test.specification.total_questions,
                "duration_minutes": generated_test.specification.duration_minutes,
                "subject_distribution": generated_test.specification.subject_distribution
            },
            "questions": [
                {
                    "question_id": q.question_id,
                    "text": q.text,
                    "options": q.options,
                    "subject": q.subject,
                    "topic": q.topic,
                    "difficulty": q.difficulty,
                    "marks": q.marks,
                    "estimated_time": q.estimated_time
                }
                for q in generated_test.questions
            ],
            "start_time": generated_test.start_time.isoformat(),
            "end_time": generated_test.end_time.isoformat(),
            "adaptive_scores": generated_test.adaptive_scores,
            "total_questions": len(generated_test.questions),
            "total_marks": sum(q.marks for q in generated_test.questions)
        }
        
        # Save test to file
        test_file_path = test_generator.save_test(generated_test)
        test_response["saved_path"] = test_file_path
        
        return {"test": test_response, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error generating test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test generation failed: {str(e)}")

@app.post("/generate-standard-test")
async def generate_standard_test(
    exam_code: str,
    total_questions: int = Query(100, description="Total number of questions"),
    duration: int = Query(120, description="Test duration in minutes")
):
    """Generate standard test following official exam pattern"""
    try:
        generated_test = test_generator.generate_standard_test(exam_code, total_questions, duration)
        
        test_response = {
            "test_id": generated_test.test_id,
            "exam_code": generated_test.specification.exam_code,
            "questions": [
                {
                    "question_id": q.question_id,
                    "text": q.text,
                    "options": q.options,
                    "subject": q.subject,
                    "topic": q.topic,
                    "difficulty": q.difficulty,
                    "marks": q.marks
                }
                for q in generated_test.questions
            ],
            "total_questions": len(generated_test.questions),
            "total_marks": sum(q.marks for q in generated_test.questions),
            "duration_minutes": duration,
            "saved_path": test_generator.save_test(generated_test)
        }
        
        return {"test": test_response, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error generating standard test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Standard test generation failed: {str(e)}")

@app.post("/student-analytics")
async def get_student_analytics(request: StudentPerformanceRequest):
    """Get comprehensive student performance analytics"""
    try:
        report = analytics_engine.generate_student_report(
            request.student_id, request.days_back
        )
        
        analytics_response = {
            "report_id": report.report_id,
            "student_id": report.student_id,
            "generated_at": report.generated_at.isoformat(),
            "time_period": {
                "start": report.time_period[0].isoformat(),
                "end": report.time_period[1].isoformat()
            },
            "overall_performance": report.overall_performance,
            "subject_analysis": report.subject_analysis,
            "predictions": report.predictions,
            "recommendations": report.recommendations,
            "metadata": getattr(report, 'metadata', {})
        }
        
        # Save report
        report_file = analytics_engine.save_report(report)
        analytics_response["saved_path"] = report_file
        
        return {"analytics": analytics_response, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error generating student analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")

@app.post("/cohort-analytics")
async def get_cohort_analytics(filters: Dict = None):
    """Get cohort analytics"""
    try:
        cohort_report = analytics_engine.generate_cohort_report(filters)
        
        return {"cohort_analytics": cohort_report, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error generating cohort analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cohort analytics failed: {str(e)}")

@app.get("/student-profile/{student_id}")
async def get_student_profile(student_id: str):
    """Get student profile and recommendations"""
    try:
        profile = student_profiler.get_or_create_profile(student_id)
        recommendations = student_profiler.get_personalized_recommendations(student_id)
        
        profile_data = {
            "student_id": profile.student_id,
            "exam_preferences": profile.exam_preferences,
            "subject_strengths": profile.subject_strengths,
            "subject_weaknesses": profile.subject_weaknesses,
            "average_score": profile.average_score,
            "total_attempts": profile.total_attempts,
            "preferred_difficulty": profile.preferred_difficulty,
            "recent_performance": profile.recent_performance,
            "learning_curve": profile.learning_curve,
            "time_per_question": profile.time_per_question,
            "recommendations": recommendations
        }
        
        return {"profile": profile_data, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error fetching student profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Profile fetch failed: {str(e)}")

@app.post("/upload-data")
async def upload_exam_data(file: UploadFile = File(...), exam_code: str = Query(...)):
    """Upload exam data for processing"""
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the data (this would use actual processing logic)
        # For now, just acknowledge the upload
        
        return {
            "message": f"File {file.filename} uploaded successfully for exam {exam_code}",
            "file_path": str(file_path),
            "exam_code": exam_code,
            "status": "uploaded"
        }
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/question-bank/stats")
async def get_question_bank_stats():
    """Get question bank statistics"""
    try:
        stats = {
            "total_questions": len(question_bank.questions) if question_bank else 0,
            "subject_distribution": dict(question_bank.subject_distribution) if question_bank else {},
            "difficulty_distribution": dict(question_bank.difficulty_distribution) if question_bank else {},
            "available_exams": list(set(q.exam_code for q in question_bank.questions)) if question_bank else []
        }
        
        return {"question_bank_stats": stats, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error fetching question bank stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats fetch failed: {str(e)}")

# Helper functions
def rule_based_classification(text: str, options: List[str]) -> Dict:
    """Simple rule-based question classification for demo"""
    text_lower = text.lower()
    
    # Subject classification based on keywords
    subject_keywords = {
        'mathematics': ['calculate', 'value', 'sum', 'product', 'equation', 'solve', 'formula', 'number', 'angle', 'area'],
        'english': ['meaning', 'synonym', 'antonym', 'passage', 'grammar', 'comprehension', 'word', 'sentence'],
        'reasoning': ['logical', 'sequence', 'pattern', 'arrangement', 'classification', 'analogy', 'series'],
        'general_awareness': ['current', 'affairs', 'recent', 'important', 'event', 'news', 'awareness'],
        'geography': ['river', 'mountain', 'country', 'capital', 'latitude', 'longitude', 'climate'],
        'history': ['ancient', 'medieval', 'modern', 'century', 'year', 'war', 'empire', 'king'],
        'science': ['element', 'atom', 'molecule', 'reaction', 'formula', 'experiment', 'theory']
    }
    
    subject_scores = {}
    for subject, keywords in subject_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            subject_scores[subject] = score
    
    predicted_subject = max(subject_scores, key=subject_scores.get) if subject_scores else 'general'
    subject_confidence = min(subject_scores.get(predicted_subject, 0) / 3, 1.0)
    
    # Difficulty classification
    difficulty_indicators = {
        'easy': ['basic', 'simple', 'elementary', 'general', 'common'],
        'medium': ['calculate', 'determine', 'find', 'solve', 'explain'],
        'hard': ['analyze', 'evaluate', 'synthesize', 'derive', 'complex']
    }
    
    difficulty_scores = {}
    for difficulty, indicators in difficulty_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        if score > 0:
            difficulty_scores[difficulty] = score
    
    predicted_difficulty = max(difficulty_scores, key=difficulty_scores.get) if difficulty_scores else 'medium'
    difficulty_confidence = min(difficulty_scores.get(predicted_difficulty, 0) / 2, 1.0)
    
    return {
        "predicted_subject": predicted_subject,
        "subject_confidence": subject_confidence,
        "predicted_topic": f"{predicted_subject}_general",
        "topic_confidence": subject_confidence * 0.8,
        "predicted_difficulty": predicted_difficulty,
        "difficulty_confidence": difficulty_confidence,
        "subject_probabilities": {k: v/sum(subject_scores.values()) for k, v in subject_scores.items()},
        "topic_probabilities": {f"{predicted_subject}_general": 1.0},
        "difficulty_probabilities": {k: v/sum(difficulty_scores.values()) if difficulty_scores else 0.33 for k, v in difficulty_scores.items()}
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "status": "error"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)