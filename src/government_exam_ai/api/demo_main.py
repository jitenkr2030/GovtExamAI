"""
Simplified FastAPI Backend for Government Exam AI System
Demo version with basic functionality to demonstrate the system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Government Exam AI System - Demo",
    description="AI-powered platform for 150+ government exam preparation (Demo Version)",
    version="1.0.0-demo"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo data
EXAM_DATA = {
    "upsc_cse": {
        "name": "UPSC Civil Services Examination",
        "full_name": "Union Public Service Commission Civil Services",
        "stages": ["prelims", "mains", "interview"],
        "subjects": ["history", "geography", "polity", "economy", "science", "current_affairs", "essay", "gs"],
        "difficulty": "very_high"
    },
    "ssc_cgl": {
        "name": "SSC Combined Graduate Level",
        "full_name": "Staff Selection Commission Combined Graduate Level",
        "stages": ["tier1", "tier2", "tier3", "tier4"],
        "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
        "difficulty": "medium"
    },
    "ibps_po": {
        "name": "IBPS Probationary Officer",
        "full_name": "Institute of Banking Personnel Selection PO",
        "stages": ["prelims", "mains", "interview"],
        "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge", "professional_knowledge"],
        "difficulty": "high"
    },
    "ssc_chsl": {
        "name": "SSC Combined Higher Secondary Level",
        "full_name": "Staff Selection Commission Combined Higher Secondary",
        "stages": ["tier1", "tier2", "tier3"],
        "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
        "difficulty": "low"
    },
    "rrb_ntpc": {
        "name": "RRB Non-Technical Popular Categories",
        "full_name": "Railway Recruitment Board NTPC",
        "stages": ["stage1", "stage2"],
        "subjects": ["general_awareness", "mathematics", "general_intelligence", "reasoning"],
        "difficulty": "medium"
    }
}

SAMPLE_QUESTIONS = [
    {
        "id": "Q001",
        "text": "What is the capital of France?",
        "options": ["London", "Berlin", "Paris", "Madrid"],
        "correct_answer": "C",
        "subject": "geography",
        "difficulty": "easy"
    },
    {
        "id": "Q002", 
        "text": "Calculate: 15 × 8 ÷ 4 + 12",
        "options": ["42", "39", "45", "48"],
        "correct_answer": "A",
        "subject": "mathematics",
        "difficulty": "medium"
    },
    {
        "id": "Q003",
        "text": "Choose the word most opposite to 'OBSTINATE'",
        "options": ["Stubborn", "Flexible", "Persistent", "Determined"],
        "correct_answer": "B",
        "subject": "english",
        "difficulty": "medium"
    }
]

STUDENT_PROFILES = {
    "demo_student": {
        "student_id": "demo_student",
        "average_score": 65.5,
        "total_attempts": 12,
        "subject_strengths": {"english": 75.0, "reasoning": 68.0},
        "subject_weaknesses": {"mathematics": 45.0, "general_awareness": 52.0},
        "recent_performance": [58.2, 61.5, 64.1, 62.8, 65.3, 65.5],
        "preferred_difficulty": "medium"
    }
}

# Pydantic models
class QuestionRequest(BaseModel):
    text: str
    options: List[str] = []
    exam_code: str = "ssc_cgl"

class AnswerEvaluationRequest(BaseModel):
    question_id: str
    student_answer: str
    correct_answer: str
    answer_type: str = "objective"
    max_marks: float = 1.0

class TestGenerationRequest(BaseModel):
    student_id: str
    exam_code: str
    total_questions: int = 10
    duration_minutes: int = 60

# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Government Exam AI System - Demo Version",
        "version": "1.0.0-demo",
        "status": "operational",
        "features": {
            "exam_categories": len(EXAM_DATA),
            "question_classification": "Rule-based demo",
            "answer_evaluation": "Basic evaluation",
            "test_generation": "Sample tests",
            "analytics": "Basic analytics"
        },
        "supported_exams": list(EXAM_DATA.keys()),
        "api_docs": "/docs"
    }

@app.get("/exams")
async def get_supported_exams():
    """Get all supported government exams"""
    return {
        "exams": EXAM_DATA,
        "total_categories": len(EXAM_DATA),
        "message": "Demo version with sample exam data"
    }

@app.post("/classify-question")
async def classify_question(request: QuestionRequest):
    """Simple rule-based question classification for demo"""
    text_lower = request.text.lower()
    
    # Rule-based subject classification
    subject_keywords = {
        'mathematics': ['calculate', 'value', 'sum', 'product', 'equation', 'solve', 'formula', 'number'],
        'english': ['meaning', 'synonym', 'antonym', 'passage', 'grammar', 'comprehension', 'word'],
        'reasoning': ['logical', 'sequence', 'pattern', 'arrangement', 'classification', 'analogy'],
        'geography': ['river', 'mountain', 'country', 'capital', 'latitude', 'longitude'],
        'history': ['ancient', 'medieval', 'modern', 'century', 'war', 'empire', 'king'],
        'general_awareness': ['current', 'affairs', 'recent', 'important', 'event', 'news']
    }
    
    subject_scores = {}
    for subject, keywords in subject_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            subject_scores[subject] = score
    
    predicted_subject = max(subject_scores, key=subject_scores.get) if subject_scores else 'general'
    confidence = min(subject_scores.get(predicted_subject, 0) / 3, 1.0)
    
    # Difficulty classification
    difficulty_indicators = {
        'easy': ['basic', 'simple', 'general', 'what', 'where', 'who'],
        'medium': ['calculate', 'determine', 'find', 'solve', 'explain', 'describe'],
        'hard': ['analyze', 'evaluate', 'synthesize', 'compare', 'contrast', 'justify']
    }
    
    difficulty_scores = {}
    for difficulty, indicators in difficulty_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        if score > 0:
            difficulty_scores[difficulty] = score
    
    predicted_difficulty = max(difficulty_scores, key=difficulty_scores.get) if difficulty_scores else 'medium'
    difficulty_confidence = min(difficulty_scores.get(predicted_difficulty, 0) / 2, 1.0)
    
    return {
        "classification": {
            "predicted_subject": predicted_subject,
            "subject_confidence": confidence,
            "predicted_topic": f"{predicted_subject}_general",
            "topic_confidence": confidence * 0.8,
            "predicted_difficulty": predicted_difficulty,
            "difficulty_confidence": difficulty_confidence,
            "method": "rule-based (demo)"
        },
        "status": "success"
    }

@app.post("/evaluate-answer")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Basic answer evaluation for demo"""
    is_correct = request.student_answer.strip().upper() == request.correct_answer.strip().upper()
    
    # Simple similarity check for text answers
    similarity_score = 0.0
    if not is_correct and request.student_answer and request.correct_answer:
        # Very basic similarity (in real system, would use NLP)
        student_words = set(request.student_answer.lower().split())
        correct_words = set(request.correct_answer.lower().split())
        if student_words and correct_words:
            similarity_score = len(student_words.intersection(correct_words)) / len(student_words.union(correct_words))
    
    marks_awarded = request.max_marks if is_correct else (similarity_score * request.max_marks * 0.5)
    percentage = (marks_awarded / request.max_marks) * 100
    
    feedback = "Correct answer!" if is_correct else "Incorrect. " + ("Partially correct" if similarity_score > 0.3 else "Review the concept and try again.")
    
    return {
        "evaluation": {
            "question_id": request.question_id,
            "marks_awarded": round(marks_awarded, 2),
            "max_marks": request.max_marks,
            "percentage": round(percentage, 2),
            "feedback": feedback,
            "is_correct": is_correct,
            "similarity_score": round(similarity_score, 3),
            "method": "basic evaluation (demo)"
        },
        "status": "success"
    }

@app.post("/generate-test")
async def generate_test(request: TestGenerationRequest):
    """Generate sample test for demo"""
    # Get random questions based on exam
    selected_questions = []
    for i in range(min(request.total_questions, len(SAMPLE_QUESTIONS))):
        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)].copy()
        question["question_id"] = f"{question['id']}_{i+1}"
        selected_questions.append(question)
    
    test_id = f"demo_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "test": {
            "test_id": test_id,
            "student_id": request.student_id,
            "exam_code": request.exam_code,
            "total_questions": len(selected_questions),
            "questions": selected_questions,
            "duration_minutes": request.duration_minutes,
            "total_marks": len(selected_questions),
            "generated_at": datetime.now().isoformat(),
            "method": "sample generation (demo)"
        },
        "status": "success"
    }

@app.get("/student-profile/{student_id}")
async def get_student_profile(student_id: str):
    """Get student profile"""
    if student_id in STUDENT_PROFILES:
        profile = STUDENT_PROFILES[student_id].copy()
        profile["recommendations"] = {
            "focus_areas": list(profile["subject_weaknesses"].keys()),
            "strengths_to_leverage": list(profile["subject_strengths"].keys()),
            "study_suggestions": [
                "Practice more mathematics problems daily",
                "Read current affairs regularly",
                "Take timed practice tests"
            ]
        }
        return {"profile": profile, "status": "success"}
    else:
        # Create a default profile
        default_profile = {
            "student_id": student_id,
            "average_score": 0.0,
            "total_attempts": 0,
            "subject_strengths": {},
            "subject_weaknesses": {},
            "recent_performance": [],
            "preferred_difficulty": "medium",
            "recommendations": {
                "focus_areas": ["Start with any subject"],
                "study_suggestions": ["Take initial practice tests to assess current level"]
            }
        }
        return {"profile": default_profile, "status": "created"}

@app.post("/student-analytics")
async def get_student_analytics(student_id: str = "demo_student"):
    """Basic student analytics for demo"""
    profile = STUDENT_PROFILES.get(student_id, {})
    
    analytics = {
        "report_id": f"analytics_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "student_id": student_id,
        "generated_at": datetime.now().isoformat(),
        "overall_performance": {
            "average_score": profile.get("average_score", 0),
            "total_attempts": profile.get("total_attempts", 0),
            "trend": "improving" if len(profile.get("recent_performance", [])) > 1 and profile.get("recent_performance", [])[-1] > profile.get("recent_performance", [])[0] else "stable"
        },
        "subject_analysis": {
            "strengths": profile.get("subject_strengths", {}),
            "weaknesses": profile.get("subject_weaknesses", {}),
            "average_subject_score": 60.0  # Demo value
        },
        "predictions": {
            "next_test_score": profile.get("average_score", 0) + 2.0,
            "exam_readiness": "needs_practice" if profile.get("average_score", 0) < 60 else "ready"
        },
        "recommendations": [
            "Focus on weak subjects identified in analysis",
            "Practice time management during tests",
            "Review incorrect answers thoroughly"
        ],
        "method": "basic analytics (demo)"
    }
    
    return {"analytics": analytics, "status": "success"}

@app.get("/question-bank/stats")
async def get_question_bank_stats():
    """Get question bank statistics"""
    return {
        "question_bank_stats": {
            "total_questions": len(SAMPLE_QUESTIONS),
            "subject_distribution": {
                "mathematics": 1,
                "english": 1, 
                "geography": 1
            },
            "difficulty_distribution": {
                "easy": 1,
                "medium": 2
            },
            "available_exams": list(EXAM_DATA.keys())
        },
        "status": "success"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-demo"
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
    print("Starting Government Exam AI System Demo Server...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)