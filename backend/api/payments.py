from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime, timedelta
import razorpay
import os
import random

router = APIRouter()

# Initialize Razorpay client (mock in development)
class PaymentRequest(BaseModel):
    user_id: str
    plan_id: str
    amount: float
    currency: str = "INR"

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

# Mock Razorpay client for demo
class MockRazorpayClient:
    def __init__(self):
        self.orders = {}
        self.payments = {}
    
    def create_order(self, data):
        order_id = f"order_{random.randint(100000, 999999)}"
        self.orders[order_id] = {
            "id": order_id,
            "amount": data["amount"],
            "currency": data["currency"],
            "receipt": data.get("receipt"),
            "status": "created"
        }
        return self.orders[order_id]
    
    def verify_payment_signature(self, data):
        # Mock verification - always returns True
        return True

# Initialize mock client
razorpay_client = MockRazorpayClient()

@router.post("/create-order")
async def create_payment_order(request: PaymentRequest):
    """Create Razorpay payment order"""
    
    try:
        # Create order
        order_data = {
            "amount": int(request.amount * 100),  # Convert to paise
            "currency": request.currency,
            "receipt": f"receipt_{request.user_id}_{request.plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "notes": {
                "user_id": request.user_id,
                "plan_id": request.plan_id
            }
        }
        
        order = razorpay_client.create_order(order_data)
        
        return {
            "order_id": order["id"],
            "amount": request.amount,
            "currency": request.currency,
            "description": f"Subscription to {request.plan_id}",
            "key_id": "rzp_test_key",  # Mock key
            "status": "created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/verify-payment")
async def verify_payment(request: VerifyPaymentRequest):
    """Verify Razorpay payment"""
    
    try:
        # Verify signature
        is_valid = razorpay_client.verify_payment_signature({
            "razorpay_order_id": request.razorpay_order_id,
            "razorpay_payment_id": request.razorpay_payment_id,
            "razorpay_signature": request.razorpay_signature
        })
        
        if is_valid:
            return {
                "status": "success",
                "message": "Payment verified successfully",
                "payment_id": request.razorpay_payment_id,
                "verified_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid payment signature")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/marketplace/purchase")
async def purchase_marketplace_item(request: PaymentRequest):
    """Purchase from marketplace"""
    
    marketplace_prices = {
        "UPSC_Prelims": 599,
        "SSC_CGL": 299,
        "Banking_Complete": 399,
        "RRB_NTPC": 199,
        "State_PCS": 349,
        "Current_Affairs": 99,
        "NCERT_Summaries": 199,
        "Video_Course": 1999
    }
    
    expected_amount = marketplace_prices.get(request.plan_id)
    if not expected_amount:
        raise HTTPException(status_code=400, detail="Invalid marketplace item")
    
    if request.amount != expected_amount:
        raise HTTPException(status_code=400, detail="Incorrect amount")
    
    # Create order for marketplace item
    order_data = {
        "amount": int(request.amount * 100),
        "currency": request.currency,
        "receipt": f"marketplace_{request.user_id}_{request.plan_id}",
        "notes": {
            "type": "marketplace",
            "item_id": request.plan_id
        }
    }
    
    order = razorpay_client.create_order(order_data)
    
    return {
        "order_id": order["id"],
        "amount": request.amount,
        "item_id": request.plan_id,
        "type": "marketplace",
        "status": "created"
    }

@router.post("/micro-transaction")
async def process_micro_transaction(request: PaymentRequest):
    """Process micro-transaction payment"""
    
    micro_prices = {
        "answer_evaluation": 10,
        "concept_explainer": 5,
        "essay_correction": 20,
        "personalized_notes": 15,
        "interview_qa": 30,
        "mock_test_analysis": 25
    }
    
    expected_amount = micro_prices.get(request.plan_id)
    if not expected_amount:
        raise HTTPException(status_code=400, detail="Invalid micro-transaction")
    
    # Create order for micro-transaction
    order_data = {
        "amount": int(request.amount * 100),
        "currency": request.currency,
        "receipt": f"micro_{request.user_id}_{request.plan_id}",
        "notes": {
            "type": "micro_transaction",
            "service_id": request.plan_id
        }
    }
    
    order = razorpay_client.create_order(order_data)
    
    return {
        "order_id": order["id"],
        "amount": request.amount,
        "service_id": request.plan_id,
        "type": "micro_transaction",
        "status": "created"
    }

@router.get("/pricing")
async def get_all_pricing():
    """Get complete pricing information"""
    
    return {
        "subscription_plans": {
            "free_plan": {"price": 0, "features": ["10 daily questions", "2 mock tests"]},
            "basic_plan": {"price": 149, "features": ["Unlimited questions", "15 mock tests"]},
            "pro_plan": {"price": 399, "features": ["All features", "Unlimited tests"]},
            "ias_elite_plan": {"price": 1299, "features": ["Mentorship", "Advanced AI"]}
        },
        "marketplace": {
            "mock_tests": {
                "UPSC_Prelims": 599,
                "SSC_CGL": 299,
                "Banking_Complete": 399
            },
            "study_materials": {
                "Current_Affairs": 99,
                "NCERT_Summaries": 199,
                "Mind_Maps": 149
            },
            "video_courses": {
                "Complete_Syllabus": 1999,
                "Topic_Wise": 99,
                "Interview_Prep": 499
            }
        },
        "micro_transactions": {
            "answer_evaluation": 10,
            "concept_explainer": 5,
            "essay_correction": 20,
            "personalized_notes": 15,
            "interview_qa": 30
        },
        "exam_guarantee": {
            "price": 5999,
            "refund_percentage": 50,
            "duration_days": 365
        },
        "b2b": {
            "coaching_license": {"min": 20000, "max": 200000},
            "api_services": {"min": 2999, "max": 19999},
            "white_label": {"min": 10000, "max": 100000}
        }
    }