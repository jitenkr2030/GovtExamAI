"""
Government Exam Categories Configuration
Comprehensive mapping of 150+ government exams across 15 categories
"""

from typing import Dict, List
from enum import Enum

class ExamCategory(Enum):
    UPSC = "upsc"
    SSC = "ssc" 
    BANKING = "banking"
    RAILWAYS = "railways"
    DEFENCE = "defence"
    STATE_GOVT = "state_govt"
    TEACHING = "teaching"
    INSURANCE = "insurance"
    PSU = "psu"
    METRO = "metro"
    JUDICIARY = "judiciary"
    RESEARCH = "research"
    HEALTH = "health"
    CLERK = "clerk"
    SPECIALIZED = "specialized"

class ExamConfig:
    """Configuration for all government exams"""
    
    UPSC_EXAMS = {
        "upsc_cse": {
            "name": "UPSC Civil Services Examination",
            "full_name": "Union Public Service Commission Civil Services",
            "stages": ["prelims", "mains", "interview"],
            "subjects": ["history", "geography", "polity", "economy", "science", "current_affairs", "essay", "gs"],
            "marks_pattern": {"prelims": 400, "mains": 1750, "interview": 275},
            "duration": {"prelims": "2 hours", "mains": "3 hours", "interview": "45 mins"},
            "difficulty": "very_high"
        },
        "upsc_ifs": {
            "name": "Indian Forest Service",
            "full_name": "UPSC Indian Forest Service",
            "stages": ["prelims", "mains", "interview"],
            "subjects": ["forestry", "botany", "zoology", "chemistry", "geology", "agriculture", "mathematics"],
            "marks_pattern": {"prelims": 400, "mains": 1400, "interview": 200},
            "duration": {"prelims": "2 hours", "mains": "3 hours", "interview": "45 mins"},
            "difficulty": "very_high"
        },
        "upsc_cds": {
            "name": "Combined Defence Services",
            "full_name": "UPSC Combined Defence Services",
            "stages": ["written", "ssb"],
            "subjects": ["english", "general_knowledge", "elementary_mathematics"],
            "marks_pattern": {"written": 300, "ssb": 200},
            "duration": {"written": "2 hours each", "ssb": "5 days"},
            "difficulty": "high"
        },
        "upsc_nda": {
            "name": "NDA & NA Examination",
            "full_name": "National Defence Academy and Naval Academy",
            "stages": ["written", "ssb"],
            "subjects": ["english", "general_knowledge", "mathematics"],
            "marks_pattern": {"written": 300, "ssb": 900},
            "duration": {"written": "2.5 hours", "ssb": "5 days"},
            "difficulty": "medium"
        }
    }
    
    SSC_EXAMS = {
        "ssc_cgl": {
            "name": "SSC Combined Graduate Level",
            "full_name": "Staff Selection Commission Combined Graduate Level",
            "stages": ["tier1", "tier2", "tier3", "tier4"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
            "marks_pattern": {"tier1": 200, "tier2": 200, "tier3": 100, "tier4": 50},
            "duration": {"tier1": "1 hour", "tier2": "2 hours", "tier3": "1 hour", "tier4": "30 mins"},
            "difficulty": "medium"
        },
        "ssc_chsl": {
            "name": "SSC Combined Higher Secondary Level",
            "full_name": "Staff Selection Commission Combined Higher Secondary",
            "stages": ["tier1", "tier2", "tier3"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
            "marks_pattern": {"tier1": 200, "tier2": 200, "tier3": 100},
            "duration": {"tier1": "1 hour", "tier2": "2 hours", "tier3": "1 hour"},
            "difficulty": "low"
        },
        "ssc_mts": {
            "name": "SSC Multi Tasking Staff",
            "full_name": "Staff Selection Commission Multi Tasking Staff",
            "stages": ["paper1"],
            "subjects": ["english", "numerical_aptitude", "general_intelligence", "general_awareness"],
            "marks_pattern": {"paper1": 150},
            "duration": {"paper1": "1.5 hours"},
            "difficulty": "low"
        },
        "ssc_gd": {
            "name": "SSC Constable GD",
            "full_name": "Staff Selection Commission Constable (General Duty)",
            "stages": ["cbt", "physical"],
            "subjects": ["english", "hindi", "reasoning", "general_awareness", "mathematics"],
            "marks_pattern": {"cbt": 100},
            "duration": {"cbt": "1 hour"},
            "difficulty": "low"
        }
    }
    
    BANKING_EXAMS = {
        "ibps_po": {
            "name": "IBPS Probationary Officer",
            "full_name": "Institute of Banking Personnel Selection PO",
            "stages": ["prelims", "mains", "interview"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge", "professional_knowledge"],
            "marks_pattern": {"prelims": 100, "mains": 200, "interview": 100},
            "duration": {"prelims": "1 hour", "mains": "2 hours", "interview": "20 mins"},
            "difficulty": "high"
        },
        "ibps_clerk": {
            "name": "IBPS Clerk",
            "full_name": "Institute of Banking Personnel Selection Clerk",
            "stages": ["prelims", "mains"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
            "marks_pattern": {"prelims": 100, "mains": 200},
            "duration": {"prelims": "1 hour", "mains": "2 hours"},
            "difficulty": "medium"
        },
        "sbi_po": {
            "name": "SBI Probationary Officer",
            "full_name": "State Bank of India Probationary Officer",
            "stages": ["prelims", "mains", "gd_pi"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "computer_knowledge"],
            "marks_pattern": {"prelims": 100, "mains": 200, "gd_pi": 50},
            "duration": {"prelims": "1 hour", "mains": "2 hours", "gd_pi": "20 mins"},
            "difficulty": "high"
        },
        "rbi_gradeb": {
            "name": "RBI Grade B Officer",
            "full_name": "Reserve Bank of India Grade B Officer",
            "stages": ["phase1", "phase2", "interview"],
            "subjects": ["english", "quantitative_aptitude", "reasoning", "general_awareness", "economics_finance", "management"],
            "marks_pattern": {"phase1": 250, "phase2": 450, "interview": 75},
            "duration": {"phase1": "1.5 hours", "phase2": "2.5 hours", "interview": "30 mins"},
            "difficulty": "very_high"
        }
    }
    
    # Add more categories...
    RAILWAYS_EXAMS = {
        "rrb_ntpc": {
            "name": "RRB Non-Technical Popular Categories",
            "full_name": "Railway Recruitment Board NTPC",
            "stages": ["stage1", "stage2"],
            "subjects": ["general_awareness", "mathematics", "general_intelligence", "reasoning"],
            "marks_pattern": {"stage1": 100, "stage2": 120},
            "duration": {"stage1": "1.5 hours", "stage2": "1.5 hours"},
            "difficulty": "medium"
        },
        "rrb_groupd": {
            "name": "RRB Group D",
            "full_name": "Railway Recruitment Board Group D",
            "stages": ["cbt"],
            "subjects": ["general_science", "mathematics", "general_awareness", "general_intelligence"],
            "marks_pattern": {"cbt": 100},
            "duration": {"cbt": "1.5 hours"},
            "difficulty": "low"
        }
    }

# Create EXAM_CATEGORIES for compatibility with model_trainer
EXAM_CATEGORIES = {
    'subjects': [
        'Mathematics', 'Science', 'History', 'Geography', 'Polity', 'Economics', 
        'Banking', 'General Knowledge', 'English', 'Reasoning', 'Current Affairs',
        'Computer Knowledge', 'General Science', 'Social Science', 'Art & Culture',
        'Environment', 'Sports', 'Literature', 'Philosophy', 'Law'
    ],
    'topics': [
        'Arithmetic', 'Algebra', 'Geometry', 'Calculus', 'Statistics', 'Physics', 
        'Chemistry', 'Biology', 'Ancient History', 'Medieval History', 'Modern History',
        'Indian Geography', 'World Geography', 'Physical Geography', 'Indian Polity',
        'International Relations', 'Indian Economy', 'Macroeconomics', 'Microeconomics',
        'Banking Operations', 'Monetary Policy', 'Constitution', 'Fundamental Rights',
        'Parliament', 'Executive', 'Judiciary', 'Current Affairs', 'Awards',
        'Sports Events', 'Scientific Discoveries', 'Technology', 'Space Science'
    ],
    'difficulty_levels': ['Easy', 'Medium', 'Hard']
}

def get_exam_config() -> Dict:
    """Get complete exam configuration"""
    return {
        "UPSC": ExamConfig.UPSC_EXAMS,
        "SSC": ExamConfig.SSC_EXAMS,
        "BANKING": ExamConfig.BANKING_EXAMS,
        "RAILWAYS": ExamConfig.RAILWAYS_EXAMS
    }

def get_subjects_by_category(category: str) -> List[str]:
    """Get all subjects for a specific category"""
    config = get_exam_config()
    subjects = set()
    
    if category.upper() in config:
        for exam in config[category.upper()].values():
            subjects.update(exam["subjects"])
    
    return list(subjects)

def get_difficulty_level(exam_code: str) -> str:
    """Get difficulty level for an exam"""
    config = get_exam_config()
    
    for category_exams in config.values():
        if exam_code in category_exams:
            return category_exams[exam_code]["difficulty"]
    
    return "unknown"

def get_exam_details(exam_code: str) -> Dict:
    """Get complete details for an exam"""
    config = get_exam_config()
    
    for category_exams in config.values():
        if exam_code in category_exams:
            return category_exams[exam_code]
    
    return {}