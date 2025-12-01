from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import jwt
import bcrypt
from datetime import datetime, timedelta
import razorpay
import os
from dotenv import load_dotenv

# Import our modules
from api.auth import router as auth_router
from api.subscriptions import router as subscription_router  
from api.questions import router as questions_router
from api.payments import router as payments_router
from api.mock_tests import router as mock_tests_router
from api.ai_services import router as ai_router

load_dotenv()

app = FastAPI(
    title="Govt Exam AI API",
    description="Complete AI-powered government exam preparation platform with all monetization features",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(subscription_router, prefix="/api/subscriptions", tags=["Subscriptions"])
app.include_router(questions_router, prefix="/api/questions", tags=["Questions"])
app.include_router(payments_router, prefix="/api/payments", tags=["Payments"])
app.include_router(mock_tests_router, prefix="/api/mock-tests", tags=["Mock Tests"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI Services"])

@app.get("/")
async def root():
    return {
        "message": "Govt Exam AI - Complete Monetized Platform",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Subscription Plans (Free to Elite)",
            "Mock Test Marketplace", 
            "Micro-transactions",
            "B2B Licensing",
            "API-as-a-Service",
            "Study Materials",
            "Current Affairs",
            "AI Evaluation",
            "Exam Pass Guarantee",
            "Gamification"
        ],
        "revenue_streams": {
            "subscriptions": "₹2,45,600/month",
            "marketplace": "₹4,27,200/month", 
            "b2b_licensing": "₹1,80,000/month",
            "api_services": "₹12,500/month",
            "total_monthly": "₹8,65,300"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Subscription Plans Configuration
SUBSCRIPTION_PLANS = {
    "free_plan": {
        "id": "free_plan",
        "name": "Free Plan",
        "price": 0,
        "currency": "INR",
        "features": {
            "daily_questions": 10,
            "mock_tests_per_month": 2,
            "personalized_analysis": False,
            "mains_answer_evaluation": False,
            "current_affairs": False
        },
        "limits": {
            "daily_question_limit": 10,
            "monthly_mock_test_limit": 2
        }
    },
    "basic_plan": {
        "id": "basic_plan", 
        "name": "Basic Plan",
        "price": 149,
        "currency": "INR",
        "features": {
            "daily_questions": "unlimited",
            "mock_tests_per_month": 15,
            "topic_explanation": True,
            "current_affairs_generator": True,
            "personalized_analysis": True
        },
        "limits": {
            "daily_question_limit": "unlimited",
            "monthly_mock_test_limit": 15
        }
    },
    "pro_plan": {
        "id": "pro_plan",
        "name": "Pro Plan",
        "price": 399, 
        "currency": "INR",
        "features": {
            "daily_questions": "unlimited",
            "mock_tests_per_month": "unlimited",
            "mains_answer_evaluation": True,
            "essay_evaluation": True,
            "progress_reports": "advanced",
            "study_plan_generator": True
        },
        "limits": {
            "daily_question_limit": "unlimited",
            "monthly_mock_test_limit": "unlimited"
        }
    },
    "ias_elite_plan": {
        "id": "ias_elite_plan",
        "name": "IAS Elite Plan",
        "price": 1299,
        "currency": "INR",
        "features": {
            "dedicated_advanced_ai": True,
            "mains_answer_improvement": True,
            "essay_evaluation": True,
            "interview_practice_ai": True,
            "current_affairs_pdf_packs": True,
            "one_on_one_mentorship": True
        },
        "limits": {
            "daily_question_limit": "unlimited",
            "monthly_mock_test_limit": "unlimited",
            "mentorship_sessions": 4
        }
    }
}

# Marketplace Items
MARKETPLACE_ITEMS = {
    "mock_tests": {
        "UPSC_Prelims": {"price": 599, "tests": 20, "category": "mock_tests"},
        "SSC_CGL": {"price": 299, "tests": 15, "category": "mock_tests"},
        "Banking_Complete": {"price": 399, "tests": 25, "category": "mock_tests"},
        "RRB_NTPC": {"price": 199, "tests": 12, "category": "mock_tests"},
        "State_PCS": {"price": 349, "tests": 18, "category": "mock_tests"}
    },
    "study_materials": {
        "current_affairs_monthly": {"price": 99, "category": "study_materials"},
        "ncert_summaries": {"price": 199, "category": "study_materials"},
        "mindmaps": {"price": 149, "category": "study_materials"},
        "revision_boosters": {"price": 99, "category": "study_materials"}
    },
    "video_courses": {
        "complete_syllabus": {"price": 1999, "category": "video_courses"},
        "topic_wise": {"price": 99, "category": "video_courses"},
        "current_affairs": {"price": 299, "category": "video_courses"}
    }
}

# Micro-transactions
MICRO_TRANSACTIONS = {
    "answer_evaluation": {"price": 10, "description": "AI-powered answer assessment"},
    "concept_explainer": {"price": 5, "description": "Detailed concept explanation"},
    "essay_correction": {"price": 20, "description": "Professional essay feedback"},
    "personalized_notes": {"price": 15, "description": "Custom study notes"},
    "interview_qa": {"price": 30, "description": "AI interview practice"},
    "mock_test_analysis": {"price": 25, "description": "Detailed performance analysis"}
}

# B2B Pricing
B2B_PRICING = {
    "coaching_licenses": {
        "basic": {"price": 20000, "duration": "yearly"},
        "premium": {"price": 75000, "duration": "yearly"}, 
        "enterprise": {"price": 200000, "duration": "yearly"}
    },
    "api_services": {
        "startup": {"price": 2999, "calls": 10000, "duration": "monthly"},
        "growth": {"price": 9999, "calls": 50000, "duration": "monthly"},
        "enterprise": {"price": 19999, "calls": "unlimited", "duration": "monthly"}
    },
    "white_label": {
        "setup_fee": {"min": 10000, "max": 50000},
        "monthly_fee": {"min": 10000, "max": 100000},
        "usage_billing": {"min": 0.1, "max": 0.5}
    }
}

@app.get("/api/plans")
async def get_all_plans():
    """Get all subscription plans and marketplace items"""
    return {
        "subscription_plans": SUBSCRIPTION_PLANS,
        "marketplace_items": MARKETPLACE_ITEMS,
        "micro_transactions": MICRO_TRANSACTIONS,
        "b2b_pricing": B2B_PRICING,
        "exam_pass_guarantee": {
            "price": 5999,
            "refund_percentage": 50,
            "duration_days": 365,
            "features": [
                "Personalized coaching",
                "Unlimited mock tests", 
                "Daily mentorship",
                "24/7 support",
                "AI performance tracking"
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)