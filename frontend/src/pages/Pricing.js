import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaCheck, FaCrown, FaRocket, FaCreditCard } from 'react-icons/fa';

const Pricing = () => {
  const [billingCycle, setBillingCycle] = useState('monthly');

  const plans = [
    {
      name: "Free Plan",
      icon: <FaRocket />,
      price: { monthly: 0, yearly: 0 },
      description: "Perfect for getting started",
      features: [
        "10 daily questions",
        "2 mock tests per month",
        "Basic progress tracking",
        "Limited AI analysis",
        "Community support"
      ],
      buttonText: "Get Started",
      popular: false,
      color: "#6c757d"
    },
    {
      name: "Basic Plan",
      icon: <FaCheck />,
      price: { monthly: 149, yearly: 1490 },
      description: "Best for regular practice",
      features: [
        "Unlimited questions",
        "15 mock tests per month", 
        "Topic-wise explanations",
        "Current affairs generator",
        "AI performance analysis",
        "Study plan recommendations",
        "Email support"
      ],
      buttonText: "Start Basic",
      popular: true,
      color: "#007bff"
    },
    {
      name: "Pro Plan",
      icon: <FaCrown />,
      price: { monthly: 399, yearly: 3990 },
      description: "For serious aspirants",
      features: [
        "All basic features",
        "Unlimited mock tests",
        "Mains answer evaluation",
        "Essay writing feedback", 
        "Advanced AI insights",
        "Personalized study plans",
        "Progress reports",
        "Priority support"
      ],
      buttonText: "Go Pro",
      popular: false,
      color: "#ffc107"
    },
    {
      name: "IAS Elite Plan",
      icon: <FaCrown />,
      price: { monthly: 1299, yearly: 12990 },
      description: "Complete exam preparation",
      features: [
        "All pro features",
        "Dedicated advanced AI model",
        "One-on-one mentorship sessions",
        "Interview practice AI",
        "Current affairs PDF packs",
        "Early access to new features",
        "24/7 priority support",
        "Guarantee program eligible"
      ],
      buttonText: "Get Elite",
      popular: false,
      color: "#dc3545"
    }
  ];

  const marketplaceItems = [
    {
      category: "Mock Test Series",
      items: [
        { name: "UPSC Prelims Complete Series", price: 599, tests: 20, description: "20 full-length tests with AI evaluation" },
        { name: "SSC CGL Full Length Tests", price: 299, tests: 15, description: "15 comprehensive tests for SSC CGL" },
        { name: "Banking Pre + Mains Tests", price: 399, tests: 25, description: "Complete banking exam preparation" },
        { name: "RRB NTPC Complete Series", price: 199, tests: 12, description: "Railway recruitment tests" },
        { name: "State PCS Mock Tests", price: 349, tests: 18, description: "State civil services preparation" }
      ]
    },
    {
      category: "Study Materials",
      items: [
        { name: "Monthly Current Affairs PDF", price: 99, description: "Comprehensive current affairs compilation" },
        { name: "NCERT Chapter Summaries", price: 199, description: "Complete NCERT summaries for UPSC" },
        { name: "Mind Maps Collection", price: 149, description: "Visual learning mind maps" },
        { name: "Revision Booster Tests", price: 99, description: "Quick revision tests" },
        { name: "Previous Year Papers with Solutions", price: 249, description: "Solved previous year questions" }
      ]
    },
    {
      category: "Video Courses", 
      items: [
        { name: "Complete Syllabus Coverage", price: 1999, description: "Full course videos for all subjects" },
        { name: "Topic-wise Video Lessons", price: 99, description: "Individual topic deep dives" },
        { name: "Current Affairs Videos", price: 299, description: "Monthly current affairs videos" },
        { name: "Interview Preparation", price: 499, description: "Mock interview sessions" },
        { name: "Essay Writing Course", price: 399, description: "Complete essay writing training" }
      ]
    }
  ];

  const microTransactions = [
    { name: "Answer Evaluation", price: 10, description: "AI-powered answer assessment" },
    { name: "Concept Explainer", price: 5, description: "Detailed concept explanation" },
    { name: "Essay Correction", price: 20, description: "Professional essay feedback" },
    { name: "Personalized Notes", price: 15, description: "Custom study notes" },
    { name: "Interview Q&A Session", price: 30, description: "AI interview practice" },
    { name: "Mock Test Analysis", price: 25, description: "Detailed performance analysis" }
  ];

  return (
    <div className="pricing-page">
      <div className="container">
        <motion.div 
          className="pricing-header"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>Choose Your Learning Plan</h1>
          <p>Flexible pricing for every budget and requirement</p>
          
          <div className="billing-toggle">
            <button 
              className={billingCycle === 'monthly' ? 'active' : ''}
              onClick={() => setBillingCycle('monthly')}
            >
              Monthly
            </button>
            <button 
              className={billingCycle === 'yearly' ? 'active' : ''}
              onClick={() => setBillingCycle('yearly')}
            >
              Yearly <span className="discount">Save 17%</span>
            </button>
          </div>
        </motion.div>

        {/* Subscription Plans */}
        <div className="pricing-grid">
          {plans.map((plan, index) => (
            <motion.div 
              key={index}
              className={`pricing-card ${plan.popular ? 'popular' : ''}`}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              {plan.popular && <div className="popular-badge">Most Popular</div>}
              
              <div className="plan-header">
                <div className="plan-icon" style={{ color: plan.color }}>
                  {plan.icon}
                </div>
                <h3>{plan.name}</h3>
                <p>{plan.description}</p>
              </div>
              
              <div className="plan-price">
                <span className="currency">₹</span>
                <span className="amount">
                  {billingCycle === 'monthly' ? plan.price.monthly : Math.floor(plan.price.yearly / 12)}
                </span>
                <span className="period">/{billingCycle === 'monthly' ? 'month' : 'month'}</span>
              </div>
              
              <div className="plan-features">
                {plan.features.map((feature, idx) => (
                  <div key={idx} className="feature-item">
                    <FaCheck className="check-icon" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>
              
              <button className="btn btn-primary btn-block">
                {plan.buttonText}
              </button>
            </motion.div>
          ))}
        </div>

        {/* Exam Pass Guarantee */}
        <div className="guarantee-section">
          <div className="guarantee-card">
            <div className="guarantee-badge">🔒 GUARANTEED</div>
            <h3>Exam Pass Guarantee Program</h3>
            <p>Clear your preliminary exam within 1 year or get 50% refund</p>
            
            <div className="guarantee-features">
              <div>✅ Personalized coaching</div>
              <div>✅ Unlimited mock tests</div>
              <div>✅ Daily mentorship sessions</div>
              <div>✅ 24/7 priority support</div>
              <div>✅ AI-powered performance tracking</div>
              <div>✅ Custom study plans</div>
            </div>
            
            <div className="guarantee-price">
              <span className="original-price">₹7,999</span>
              <span className="current-price">₹5,999</span>
              <span className="period">/year</span>
            </div>
            
            <button className="btn btn-success btn-large">
              Get Guarantee Plan
            </button>
          </div>
        </div>

        {/* Marketplace */}
        <div className="marketplace-section">
          <h2>Additional Products & Services</h2>
          <div className="marketplace-grid">
            {marketplaceItems.map((category, index) => (
              <div key={index} className="category-card">
                <h3>{category.category}</h3>
                <div className="items-list">
                  {category.items.map((item, idx) => (
                    <div key={idx} className="item">
                      <div className="item-info">
                        <span className="item-name">{item.name}</span>
                        <span className="item-description">{item.description}</span>
                        {item.tests && <span className="item-details">{item.tests} tests</span>}
                      </div>
                      <div className="item-price">₹{item.price}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Micro-transactions */}
        <div className="micro-transactions-section">
          <h2>Pay-Per-Use Features</h2>
          <div className="micro-grid">
            {microTransactions.map((transaction, index) => (
              <div key={index} className="micro-item">
                <div className="micro-info">
                  <h4>{transaction.name}</h4>
                  <p>{transaction.description}</p>
                </div>
                <div className="micro-price">₹{transaction.price}</div>
                <button className="btn btn-outline btn-small">
                  Buy Now
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* B2B Section */}
        <div className="b2b-section">
          <h2>B2B & Enterprise Solutions</h2>
          <div className="b2b-grid">
            <div className="b2b-card">
              <h3>Coaching Institute License</h3>
              <ul>
                <li>API Access for your students</li>
                <li>Custom branding</li>
                <li>Performance analytics dashboard</li>
                <li>Unlimited mock test generation</li>
              </ul>
              <div className="b2b-pricing">
                <span>₹20,000 - ₹2,00,000/year</span>
              </div>
            </div>
            
            <div className="b2b-card">
              <h3>API-as-a-Service</h3>
              <ul>
                <li>MCQ solving API</li>
                <li>Answer evaluation API</li>
                <li>Question generation API</li>
                <li>Performance analytics API</li>
              </ul>
              <div className="b2b-pricing">
                <span>₹0.1 - ₹0.5 per API call</span>
              </div>
            </div>
            
            <div className="b2b-card">
              <h3>White Label Solution</h3>
              <ul>
                <li>Your brand, our AI</li>
                <li>Custom domain</li>
                <li>Complete feature access</li>
                <li>Dedicated support</li>
              </ul>
              <div className="b2b-pricing">
                <span>₹10,000 - ₹1,00,000/month</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pricing;