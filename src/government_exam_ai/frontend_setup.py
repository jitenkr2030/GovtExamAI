"""
Frontend Web Interface for Government Exam AI System
React-based frontend with all system features
"""

import json
from pathlib import Path
from datetime import datetime

# Create the main HTML file
html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Government Exam AI System</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        
        .nav-tab {
            flex: 1;
            padding: 15px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            color: #6c757d;
            transition: all 0.3s ease;
        }
        
        .nav-tab.active {
            background: white;
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
        }
        
        .nav-tab:hover {
            background: #e9ecef;
        }
        
        .content {
            padding: 30px;
            min-height: 600px;
        }
        
        .section {
            display: none;
        }
        
        .section.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus,
        .form-group textarea:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-group textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .result-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .card {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        .card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 2px;
        }
        
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .question-item {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .question-text {
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .options {
            margin-left: 20px;
        }
        
        .option {
            display: block;
            margin: 5px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect } = React;
        
        // API Base URL
        const API_BASE = 'http://localhost:8000';
        
        // Main App Component
        function App() {
            const [activeTab, setActiveTab] = useState('exams');
            const [loading, setLoading] = useState(false);
            const [result, setResult] = useState(null);
            const [error, setError] = useState(null);
            
            const tabs = [
                { id: 'exams', label: 'Exam Categories', icon: '📚' },
                { id: 'classify', label: 'Question Classification', icon: '🔍' },
                { id: 'evaluate', label: 'Answer Evaluation', icon: '✅' },
                { id: 'generate', label: 'Test Generation', icon: '📝' },
                { id: 'analytics', label: 'Analytics', icon: '📊' }
            ];
            
            return (
                <div className="container">
                    <div className="header">
                        <h1>Government Exam AI System</h1>
                        <p>AI-Powered Platform for 150+ Government Exam Preparation</p>
                    </div>
                    
                    <div className="nav-tabs">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                {tab.icon} {tab.label}
                            </button>
                        ))}
                    </div>
                    
                    <div className="content">
                        {error && <div className="error">{error}</div>}
                        
                        {activeTab === 'exams' && <ExamCategories />}
                        {activeTab === 'classify' && <QuestionClassification onResult={setResult} />}
                        {activeTab === 'evaluate' && <AnswerEvaluation onResult={setResult} />}
                        {activeTab === 'generate' && <TestGeneration onResult={setResult} />}
                        {activeTab === 'analytics' && <AnalyticsDashboard onResult={setResult} />}
                        
                        {result && (
                            <div className="result-card">
                                <h3>Latest Result</h3>
                                <pre>{JSON.stringify(result, null, 2)}</pre>
                            </div>
                        )}
                    </div>
                </div>
            );
        }
        
        // Exam Categories Component
        function ExamCategories() {
            const [exams, setExams] = useState({});
            const [loading, setLoading] = useState(true);
            
            useEffect(() => {
                fetchExams();
            }, []);
            
            const fetchExams = async () => {
                try {
                    const response = await fetch(`${API_BASE}/exams`);
                    const data = await response.json();
                    setExams(data.exams || {});
                } catch (err) {
                    console.error('Error fetching exams:', err);
                } finally {
                    setLoading(false);
                }
            };
            
            if (loading) {
                return <div className="loading"><div className="spinner"></div>Loading exam categories...</div>;
            }
            
            return (
                <div>
                    <h2>Supported Government Exams</h2>
                    <p>Comprehensive coverage of 150+ government exams across 15 categories</p>
                    
                    <div className="grid">
                        {Object.entries(exams).map(([category, examList]) => (
                            <div key={category} className="card">
                                <h4>{category} Exams</h4>
                                {examList.map(exam => (
                                    <div key={exam.code} style={{marginBottom: '15px', paddingBottom: '10px', borderBottom: '1px solid #eee'}}>
                                        <strong>{exam.name}</strong>
                                        <div style={{fontSize: '0.9em', color: '#666', marginTop: '5px'}}>
                                            <div>Code: {exam.code}</div>
                                            <div>Difficulty: <span className={`badge badge-${getDifficultyColor(exam.difficulty)}`}>{exam.difficulty}</span></div>
                                            <div>Stages: {exam.stages.join(', ')}</div>
                                            <div>Total Marks: {exam.total_marks}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            );
        }
        
        // Question Classification Component
        function QuestionClassification({ onResult }) {
            const [formData, setFormData] = useState({
                text: '',
                options: ['', '', '', ''],
                correct_answer: '',
                exam_code: 'ssc_cgl'
            });
            
            const handleSubmit = async (e) => {
                e.preventDefault();
                try {
                    const response = await fetch(`${API_BASE}/classify-question`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(formData)
                    });
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Classification error:', err);
                }
            };
            
            const handleOptionChange = (index, value) => {
                const newOptions = [...formData.options];
                newOptions[index] = value;
                setFormData({...formData, options: newOptions});
            };
            
            return (
                <div>
                    <h2>Question Classification</h2>
                    <p>AI-powered classification of questions by subject, topic, and difficulty</p>
                    
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Question Text:</label>
                            <textarea
                                value={formData.text}
                                onChange={(e) => setFormData({...formData, text: e.target.value})}
                                placeholder="Enter your question here..."
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Options (Optional):</label>
                            {formData.options.map((option, index) => (
                                <input
                                    key={index}
                                    type="text"
                                    value={option}
                                    onChange={(e) => handleOptionChange(index, e.target.value)}
                                    placeholder={`Option ${String.fromCharCode(65 + index)}`}
                                />
                            ))}
                        </div>
                        
                        <div className="form-group">
                            <label>Exam Code:</label>
                            <select
                                value={formData.exam_code}
                                onChange={(e) => setFormData({...formData, exam_code: e.target.value})}
                            >
                                <option value="ssc_cgl">SSC CGL</option>
                                <option value="ibps_po">IBPS PO</option>
                                <option value="upsc_cse">UPSC CSE</option>
                                <option value="ssc_chsl">SSC CHSL</option>
                                <option value="rrb_ntpc">RRB NTPC</option>
                            </select>
                        </div>
                        
                        <button type="submit" className="btn">Classify Question</button>
                    </form>
                </div>
            );
        }
        
        // Answer Evaluation Component
        function AnswerEvaluation({ onResult }) {
            const [formData, setFormData] = useState({
                question_id: '',
                student_answer: '',
                correct_answer: '',
                answer_type: 'objective',
                max_marks: 1.0,
                subject: '',
                keywords: ''
            });
            
            const handleSubmit = async (e) => {
                e.preventDefault();
                const keywords = formData.keywords.split(',').map(k => k.trim()).filter(k => k);
                
                try {
                    const response = await fetch(`${API_BASE}/evaluate-answer`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            ...formData,
                            keywords
                        })
                    });
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Evaluation error:', err);
                }
            };
            
            return (
                <div>
                    <h2>Answer Evaluation</h2>
                    <p>AI-powered evaluation of student answers with detailed feedback</p>
                    
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Question ID:</label>
                            <input
                                type="text"
                                value={formData.question_id}
                                onChange={(e) => setFormData({...formData, question_id: e.target.value})}
                                placeholder="e.g., Q1, Q2..."
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Student Answer:</label>
                            <textarea
                                value={formData.student_answer}
                                onChange={(e) => setFormData({...formData, student_answer: e.target.value})}
                                placeholder="Enter student answer here..."
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Correct Answer:</label>
                            <input
                                type="text"
                                value={formData.correct_answer}
                                onChange={(e) => setFormData({...formData, correct_answer: e.target.value})}
                                placeholder="Correct answer"
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Answer Type:</label>
                            <select
                                value={formData.answer_type}
                                onChange={(e) => setFormData({...formData, answer_type: e.target.value})}
                            >
                                <option value="objective">Objective (MCQ)</option>
                                <option value="subjective">Subjective (Descriptive)</option>
                                <option value="essay">Essay</option>
                            </select>
                        </div>
                        
                        <div className="form-group">
                            <label>Maximum Marks:</label>
                            <input
                                type="number"
                                step="0.1"
                                value={formData.max_marks}
                                onChange={(e) => setFormData({...formData, max_marks: parseFloat(e.target.value)})}
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Subject (Optional):</label>
                            <input
                                type="text"
                                value={formData.subject}
                                onChange={(e) => setFormData({...formData, subject: e.target.value})}
                                placeholder="e.g., mathematics, english"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Keywords (comma-separated, Optional):</label>
                            <input
                                type="text"
                                value={formData.keywords}
                                onChange={(e) => setFormData({...formData, keywords: e.target.value})}
                                placeholder="keyword1, keyword2, keyword3"
                            />
                        </div>
                        
                        <button type="submit" className="btn">Evaluate Answer</button>
                    </form>
                </div>
            );
        }
        
        // Test Generation Component
        function TestGeneration({ onResult }) {
            const [formData, setFormData] = useState({
                student_id: 'demo_student',
                exam_code: 'ssc_cgl',
                total_questions: 50,
                duration_minutes: 60,
                adaptive: true,
                target_score: 75.0
            });
            
            const handleSubmit = async (e) => {
                e.preventDefault();
                try {
                    const response = await fetch(`${API_BASE}/generate-test`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(formData)
                    });
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Test generation error:', err);
                }
            };
            
            const generateStandardTest = async () => {
                try {
                    const response = await fetch(`${API_BASE}/generate-standard-test?exam_code=${formData.exam_code}&total_questions=${formData.total_questions}&duration=${formData.duration_minutes}`);
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Standard test generation error:', err);
                }
            };
            
            return (
                <div>
                    <h2>Test Generation</h2>
                    <p>Generate adaptive mock tests based on student performance and exam patterns</p>
                    
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Student ID:</label>
                            <input
                                type="text"
                                value={formData.student_id}
                                onChange={(e) => setFormData({...formData, student_id: e.target.value})}
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Exam:</label>
                            <select
                                value={formData.exam_code}
                                onChange={(e) => setFormData({...formData, exam_code: e.target.value})}
                            >
                                <option value="ssc_cgl">SSC CGL</option>
                                <option value="ibps_po">IBPS PO</option>
                                <option value="upsc_cse">UPSC CSE</option>
                                <option value="ssc_chsl">SSC CHSL</option>
                                <option value="rrb_ntpc">RRB NTPC</option>
                            </select>
                        </div>
                        
                        <div className="form-group">
                            <label>Total Questions:</label>
                            <input
                                type="number"
                                value={formData.total_questions}
                                onChange={(e) => setFormData({...formData, total_questions: parseInt(e.target.value)})}
                                min="10"
                                max="200"
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Duration (minutes):</label>
                            <input
                                type="number"
                                value={formData.duration_minutes}
                                onChange={(e) => setFormData({...formData, duration_minutes: parseInt(e.target.value)})}
                                min="30"
                                max="300"
                                required
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={formData.adaptive}
                                    onChange={(e) => setFormData({...formData, adaptive: e.target.checked})}
                                />
                                Adaptive Test (based on student performance)
                            </label>
                        </div>
                        
                        <div className="form-group">
                            <label>Target Score (%):</label>
                            <input
                                type="number"
                                step="0.1"
                                value={formData.target_score}
                                onChange={(e) => setFormData({...formData, target_score: parseFloat(e.target.value)})}
                                min="0"
                                max="100"
                            />
                        </div>
                        
                        <button type="submit" className="btn">Generate Adaptive Test</button>
                        <button type="button" className="btn" style={{marginLeft: '10px'}} onClick={generateStandardTest}>
                            Generate Standard Test
                        </button>
                    </form>
                </div>
            );
        }
        
        // Analytics Dashboard Component
        function AnalyticsDashboard({ onResult }) {
            const [studentId, setStudentId] = useState('demo_student');
            const [daysBack, setDaysBack] = useState(30);
            const [profile, setProfile] = useState(null);
            
            const getStudentAnalytics = async () => {
                try {
                    const response = await fetch(`${API_BASE}/student-analytics`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            student_id: studentId,
                            days_back: daysBack
                        })
                    });
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Analytics error:', err);
                }
            };
            
            const getStudentProfile = async () => {
                try {
                    const response = await fetch(`${API_BASE}/student-profile/${studentId}`);
                    const data = await response.json();
                    setProfile(data.profile);
                    onResult(data);
                } catch (err) {
                    console.error('Profile error:', err);
                }
            };
            
            const getCohortAnalytics = async () => {
                try {
                    const response = await fetch(`${API_BASE}/cohort-analytics`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({})
                    });
                    const data = await response.json();
                    onResult(data);
                } catch (err) {
                    console.error('Cohort analytics error:', err);
                }
            };
            
            return (
                <div>
                    <h2>Analytics Dashboard</h2>
                    <p>Comprehensive performance analytics and insights</p>
                    
                    <div className="card">
                        <h3>Student Performance</h3>
                        <div className="form-group">
                            <label>Student ID:</label>
                            <input
                                type="text"
                                value={studentId}
                                onChange={(e) => setStudentId(e.target.value)}
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Days to Analyze:</label>
                            <input
                                type="number"
                                value={daysBack}
                                onChange={(e) => setDaysBack(parseInt(e.target.value))}
                                min="7"
                                max="365"
                            />
                        </div>
                        
                        <button className="btn" onClick={getStudentAnalytics}>Get Student Analytics</button>
                        <button className="btn" style={{marginLeft: '10px'}} onClick={getStudentProfile}>Get Student Profile</button>
                    </div>
                    
                    {profile && (
                        <div className="result-card">
                            <h3>Student Profile</h3>
                            <div className="stats-grid">
                                <div className="stat-card">
                                    <div className="stat-value">{profile.average_score?.toFixed(1) || 'N/A'}</div>
                                    <div className="stat-label">Average Score</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{profile.total_attempts || 0}</div>
                                    <div className="stat-label">Total Attempts</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{profile.preferred_difficulty || 'medium'}</div>
                                    <div className="stat-label">Preferred Difficulty</div>
                                </div>
                            </div>
                            
                            {profile.subject_strengths && Object.keys(profile.subject_strengths).length > 0 && (
                                <div style={{marginTop: '20px'}}>
                                    <h4>Subject Strengths</h4>
                                    {Object.entries(profile.subject_strengths).map(([subject, score]) => (
                                        <span key={subject} className="badge badge-success">
                                            {subject}: {score.toFixed(1)}%
                                        </span>
                                    ))}
                                </div>
                            )}
                            
                            {profile.subject_weaknesses && Object.keys(profile.subject_weaknesses).length > 0 && (
                                <div style={{marginTop: '20px'}}>
                                    <h4>Areas for Improvement</h4>
                                    {Object.entries(profile.subject_weaknesses).map(([subject, score]) => (
                                        <span key={subject} className="badge badge-danger">
                                            {subject}: {score.toFixed(1)}%
                                        </span>
                                    ))}
                                </div>
                            )}
                            
                            {profile.recommendations && (
                                <div style={{marginTop: '20px'}}>
                                    <h4>Recommendations</h4>
                                    <ul>
                                        {Object.entries(profile.recommendations).map(([key, value]) => (
                                            <li key={key}><strong>{key}:</strong> {JSON.stringify(value)}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                    
                    <div className="card" style={{marginTop: '20px'}}>
                        <h3>Cohort Analysis</h3>
                        <p>Analyze performance across multiple students</p>
                        <button className="btn" onClick={getCohortAnalytics}>Get Cohort Analytics</button>
                    </div>
                </div>
            );
        }
        
        // Helper functions
        function getDifficultyColor(difficulty) {
            switch(difficulty?.toLowerCase()) {
                case 'easy':
                case 'low':
                    return 'success';
                case 'medium':
                case 'moderate':
                    return 'warning';
                case 'hard':
                case 'high':
                    return 'danger';
                case 'very_high':
                    return 'danger';
                default:
                    return 'info';
            }
        }
        
        // Render the app
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>'''

# Create frontend directory and save the HTML file
frontend_dir = Path("/workspace/government_exam_ai/frontend")
frontend_dir.mkdir(exist_ok=True)

with open(frontend_dir / "index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Frontend HTML file created successfully!")

# Create some sample data files for demonstration
sample_data_dir = Path("/workspace/government_exam_ai/data")
sample_data_dir.mkdir(exist_ok=True)

# Create sample exam data
sample_exam_data = [
    {
        "question": {
            "question_id": "Q001",
            "exam_code": "ssc_cgl",
            "question_text": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct_answer": "C",
            "explanation": "Paris is the capital and most populous city of France.",
            "subject": "geography",
            "topic": "countries_capitals",
            "difficulty": "easy",
            "year": 2023,
            "stage": "tier1",
            "source": "sample_data",
            "tags": ["capital", "france", "europe"]
        },
        "features": {
            "marks": 1.0,
            "estimated_time": 1.5,
            "discriminative_power": 0.7,
            "question_length": 30,
            "word_count": 8,
            "subject_encoded": 9,
            "difficulty_encoded": 1
        }
    },
    {
        "question": {
            "question_id": "Q002",
            "exam_code": "ssc_cgl",
            "question_text": "Calculate the value of 15 × 8 ÷ 4 + 12",
            "options": ["42", "39", "45", "48"],
            "correct_answer": "A",
            "explanation": "15 × 8 ÷ 4 + 12 = 120 ÷ 4 + 12 = 30 + 12 = 42",
            "subject": "mathematics",
            "topic": "arithmetic",
            "difficulty": "medium",
            "year": 2023,
            "stage": "tier1",
            "source": "sample_data",
            "tags": ["arithmetic", "calculation", "bodmas"]
        },
        "features": {
            "marks": 1.0,
            "estimated_time": 2.0,
            "discriminative_power": 0.8,
            "question_length": 35,
            "word_count": 9,
            "subject_encoded": 2,
            "difficulty_encoded": 2
        }
    },
    {
        "question": {
            "question_id": "Q003",
            "exam_code": "ibps_po",
            "question_text": "Choose the word most nearly opposite in meaning to 'OBSTINATE'",
            "options": ["Stubborn", "Flexible", "Persistent", "Determined"],
            "correct_answer": "B",
            "explanation": "Obstinate means stubborn, while flexible means adaptable or willing to change.",
            "subject": "english",
            "topic": "synonyms_antonyms",
            "difficulty": "medium",
            "year": 2023,
            "stage": "prelims",
            "source": "sample_data",
            "tags": ["antonyms", "vocabulary", "opposite"]
        },
        "features": {
            "marks": 1.0,
            "estimated_time": 2.5,
            "discriminative_power": 0.6,
            "question_length": 55,
            "word_count": 12,
            "subject_encoded": 1,
            "difficulty_encoded": 2
        }
    }
]

with open(sample_data_dir / "sample_exam_data.json", "w", encoding="utf-8") as f:
    json.dump(sample_exam_data, f, indent=2, ensure_ascii=False)

print("Sample exam data created!")

# Create student profiles directory and sample profile
profiles_dir = sample_data_dir / "student_profiles"
profiles_dir.mkdir(exist_ok=True)

sample_profile = {
    "student_id": "demo_student",
    "exam_preferences": ["ssc_cgl", "ibps_po"],
    "subject_strengths": {
        "english": 75.5,
        "reasoning": 68.2
    },
    "subject_weaknesses": {
        "mathematics": 45.3,
        "general_awareness": 52.1
    },
    "average_score": 62.8,
    "total_attempts": 15,
    "recent_performance": [58.2, 61.5, 64.1, 62.8, 65.3],
    "learning_curve": ["stable", "improving", "improving"],
    "preferred_difficulty": "medium",
    "time_per_question": {
        "english": 1.8,
        "mathematics": 2.5,
        "reasoning": 2.2,
        "general_awareness": 1.5
    }
}

with open(profiles_dir / "demo_student.json", "w") as f:
    json.dump(sample_profile, f, indent=2)

print("Sample student profile created!")

print("\\nFrontend and sample data setup complete!")
print(f"Frontend files created in: {frontend_dir}")
print(f"Sample data created in: {sample_data_dir}")
print("\\nTo run the application:")
print("1. Start the FastAPI backend: cd /workspace/government_exam_ai/api && python main.py")
print("2. Open the frontend: Open frontend/index.html in a web browser")
print("3. Make sure the backend is running on localhost:8000")