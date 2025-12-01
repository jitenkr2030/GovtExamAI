from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random
from datetime import datetime

router = APIRouter()

class QuestionRequest(BaseModel):
    exam_type: str
    subject: str  
    topic: str
    count: int = 10
    difficulty: str = "Mixed"

class AnswerEvaluationRequest(BaseModel):
    question: str
    student_answer: str
    correct_answer: str
    answer_type: str = "mains"

# Mock AI question generation
def generate_questions(exam_type: str, subject: str, topic: str, count: int) -> List[Dict[str, Any]]:
    """Generate AI questions (mock implementation)"""
    
    templates = {
        "SSC_CGL": {
            "General Awareness": [
                "Who is the current {position} of {organization}?",
                "The {event} happened in which year?",
                "Which of the following is related to {topic}?",
                "The capital of {country} is ______?"
            ],
            "Reasoning": [
                "If {A} means {operation}, then what does {B} mean?",
                "Complete the series: {sequence}",
                "Find the odd one out: {options}",
                "What comes next in the pattern: {pattern}?"
            ],
            "Mathematics": [
                "What is {percentage}% of {number}?",
                "If {workers} can complete work in {days} days, then {workers2} workers will complete in how many days?",
                "The sum of {num1} and {num2} is ______?",
                "Find the value of {expression}."
            ]
        },
        "UPSC": {
            "Polity": [
                "Which article of the Indian Constitution deals with {topic}?",
                "The {institution} was established under which article?",
                "Who can amend the Constitution under {article}?"
            ],
            "History": [
                "The {event} took place during the reign of {ruler}.",
                "Which {action} was taken during {period}?",
                "Who was the {position} during {event}?"
            ],
            "Geography": [
                "The {feature} is located in which state?",
                "Which river flows through {city}?",
                "What is the climate of {region}?"
            ]
        }
    }
    
    # Default templates if specific ones not found
    default_templates = [
        "What is the importance of {topic}?",
        "Explain the concept of {topic}.",
        "Which of the following is correct about {topic}?",
        "Define {topic} in brief."
    ]
    
    questions = []
    exam_templates = templates.get(exam_type, {}).get(subject, default_templates)
    
    for i in range(count):
        template = random.choice(exam_templates)
        
        # Generate question content
        question_text = template.replace("{topic}", topic)
        
        # Generate options
        options = generate_options(topic, exam_type)
        correct_answer = random.choice(options)
        
        question = {
            "id": f"{exam_type}_{subject}_{topic}_{i+1:03d}",
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "exam_type": exam_type,
            "subject": subject,
            "topic": topic,
            "difficulty": random.choice(["Easy", "Medium", "Hard"]),
            "created_by": "AI",
            "timestamp": datetime.now().isoformat()
        }
        
        questions.append(question)
    
    return questions

def generate_options(topic: str, exam_type: str) -> List[str]:
    """Generate multiple choice options"""
    
    if "banking" in topic.lower():
        return ["RBI", "SBI", "PNB", "BOB"]
    elif "geography" in topic.lower():
        return ["India", "China", "USA", "Brazil"]
    elif "history" in topic.lower():
        return ["Ancient", "Medieval", "Modern", "Contemporary"]
    elif "mathematics" in topic.lower():
        return [10, 20, 30, 40]
    else:
        return ["Option A", "Option B", "Option C", "Option D"]

@router.post("/questions")
async def generate_ai_questions(request: QuestionRequest):
    """Generate AI-powered questions"""
    
    try:
        questions = generate_questions(
            request.exam_type,
            request.subject,
            request.topic,
            request.count
        )
        
        return {
            "status": "success",
            "questions": questions,
            "total_generated": len(questions),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_answer(request: AnswerEvaluationRequest):
    """Evaluate student answer using AI"""
    
    if request.answer_type == "mcq":
        is_correct = request.student_answer.strip().lower() == request.correct_answer.strip().lower()
        
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "max_score": 1.0,
            "feedback": "Correct answer!" if is_correct else f"Correct answer was: {request.correct_answer}",
            "evaluation_type": "mcq"
        }
    else:
        # Mock AI evaluation for descriptive answers
        word_count = len(request.student_answer.split())
        accuracy_score = random.uniform(0.6, 0.9)  # Mock accuracy
        completeness_score = min(1.0, word_count / 100)
        
        total_score = (accuracy_score * 0.4 + completeness_score * 0.3 + 0.8 * 0.3)
        
        feedback = generate_feedback(total_score)
        
        return {
            "score": round(total_score, 2),
            "max_score": 1.0,
            "accuracy_score": round(accuracy_score, 2),
            "completeness_score": round(completeness_score, 2),
            "feedback": feedback,
            "word_count": word_count,
            "evaluation_type": "descriptive"
        }

def generate_feedback(score: float) -> str:
    """Generate AI feedback based on score"""
    
    if score >= 0.8:
        return "Excellent answer! Your response is accurate and comprehensive."
    elif score >= 0.6:
        return "Good answer with room for improvement. Consider adding more details."
    elif score >= 0.4:
        return "Average answer. Focus on accuracy and completeness."
    else:
        return "Needs significant improvement. Review the topic and try again."

@router.get("/mock-test/{exam_type}")
async def create_mock_test(exam_type: str):
    """Create AI-generated mock test"""
    
    subjects = {
        "SSC_CGL": ["General Awareness", "Reasoning", "Mathematics", "English"],
        "UPSC": ["Polity", "History", "Geography", "Economics"],
        "Banking": ["General Awareness", "Reasoning", "Numerical Ability", "English"],
        "Railway": ["General Intelligence", "General Awareness", "Reasoning", "Mathematics"]
    }
    
    exam_subjects = subjects.get(exam_type, ["General Knowledge"])
    questions_per_subject = 25  # 25 questions per subject
    total_questions = len(exam_subjects) * questions_per_subject
    
    all_questions = []
    for subject in exam_subjects:
        subject_questions = generate_questions(exam_type, subject, f"{subject} - Mixed", questions_per_subject)
        all_questions.extend(subject_questions)
    
    # Shuffle questions
    random.shuffle(all_questions)
    
    mock_test = {
        "id": f"mock_test_{exam_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exam_type": exam_type,
        "subjects": exam_subjects,
        "total_questions": total_questions,
        "time_limit_minutes": 120,
        "questions": all_questions,
        "created_at": datetime.now().isoformat(),
        "created_by": "AI Mock Test Generator"
    }
    
    return mock_test

@router.get("/current-affairs")
async def get_current_affairs():
    """Generate AI current affairs"""
    
    categories = [
        "National Politics",
        "International Relations",
        "Economy & Business", 
        "Science & Technology",
        "Sports",
        "Awards & Honours"
    ]
    
    affairs = []
    for category in categories:
        for i in range(5):  # 5 affairs per category
            affair = {
                "id": f"ca_{category}_{i+1:03d}",
                "headline": f"Latest development in {category.lower().replace(' & ', ' and ')}",
                "category": category,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "importance": random.choice(["High", "Medium", "Low"]),
                "source": random.choice(["PTI", "ANI", "The Hindu", "Indian Express"])
            }
            affairs.append(affair)
    
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_affairs": len(affairs),
        "categories": categories,
        "affairs": affairs
    }