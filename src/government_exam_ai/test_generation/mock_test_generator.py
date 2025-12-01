"""
Mock Test Generation System for Government Exams
Adaptive test generation based on student performance and exam patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import random
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from config.exam_categories import ExamConfig, get_exam_config
from ml_models.question_classifier import QuestionClassifier, Question
from evaluation.answer_evaluator import AnswerEvaluationEngine, EvaluationResult

logger = logging.getLogger(__name__)

@dataclass
class TestQuestion:
    """Enhanced question structure for test generation"""
    question_id: str
    exam_code: str
    text: str
    options: List[str]
    correct_answer: str
    subject: str
    topic: str
    difficulty: str
    marks: float
    estimated_time: float
    discriminative_power: float
    source_year: int
    tags: List[str]

@dataclass
class TestSpecification:
    """Test generation specifications"""
    exam_code: str
    total_questions: int
    total_marks: float
    duration_minutes: float
    subject_distribution: Dict[str, float]
    difficulty_distribution: Dict[str, float]
    include_explanations: bool
    adaptive: bool = False
    target_score: Optional[float] = None

@dataclass
class StudentProfile:
    """Student performance profile"""
    student_id: str
    exam_preferences: List[str]
    subject_strengths: Dict[str, float]
    subject_weaknesses: Dict[str, float]
    average_score: float
    total_attempts: int
    recent_performance: List[float]
    learning_curve: List[float]
    preferred_difficulty: str
    time_per_question: Dict[str, float]

@dataclass
class GeneratedTest:
    """Complete test structure"""
    test_id: str
    student_id: str
    specification: TestSpecification
    questions: List[TestQuestion]
    start_time: datetime
    end_time: datetime
    adaptive_scores: Dict[str, float]
    metadata: Dict

class QuestionBank:
    """Manages question bank for test generation"""
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.questions = []
        self.question_index = {}
        self.subject_distribution = {}
        self.difficulty_distribution = {}
        
        self._load_questions()
        self._build_index()
    
    def _load_questions(self):
        """Load questions from processed data files"""
        if not self.data_path.exists():
            logger.warning(f"Question bank path {self.data_path} does not exist")
            return
        
        for file_path in self.data_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    question_data = item.get('question', {})
                    feature_data = item.get('features', {})
                    
                    test_question = TestQuestion(
                        question_id=question_data.get('question_id', ''),
                        exam_code=question_data.get('exam_code', ''),
                        text=question_data.get('question_text', ''),
                        options=question_data.get('options', []),
                        correct_answer=question_data.get('correct_answer', ''),
                        subject=question_data.get('subject', ''),
                        topic=question_data.get('topic', ''),
                        difficulty=question_data.get('difficulty', 'medium'),
                        marks=feature_data.get('marks', 1.0),
                        estimated_time=feature_data.get('estimated_time', 2.0),
                        discriminative_power=feature_data.get('discriminative_power', 0.5),
                        source_year=question_data.get('year', 2024),
                        tags=question_data.get('tags', [])
                    )
                    
                    self.questions.append(test_question)
                    
            except Exception as e:
                logger.error(f"Error loading questions from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.questions)} questions into question bank")
    
    def _build_index(self):
        """Build searchable indices for questions"""
        # Index by exam, subject, topic, difficulty
        for question in self.questions:
            key = f"{question.exam_code}_{question.subject}_{question.difficulty}"
            if key not in self.question_index:
                self.question_index[key] = []
            self.question_index[key].append(question)
        
        # Calculate distributions
        self.subject_distribution = Counter(q.subject for q in self.questions)
        self.difficulty_distribution = Counter(q.difficulty for q in self.questions)
    
    def get_questions_by_criteria(self, exam_code: str = None, subject: str = None, 
                                 difficulty: str = None, limit: int = None) -> List[TestQuestion]:
        """Get questions matching specific criteria"""
        filtered_questions = self.questions.copy()
        
        if exam_code:
            filtered_questions = [q for q in filtered_questions if q.exam_code == exam_code]
        
        if subject:
            filtered_questions = [q for q in filtered_questions if q.subject == subject]
        
        if difficulty:
            filtered_questions = [q for q in filtered_questions if q.difficulty == difficulty]
        
        # Sort by discriminative power (high power questions are better)
        filtered_questions.sort(key=lambda x: x.discriminative_power, reverse=True)
        
        if limit:
            filtered_questions = filtered_questions[:limit]
        
        return filtered_questions
    
    def get_random_questions(self, count: int, exam_code: str = None, 
                           subject: str = None, difficulty: str = None) -> List[TestQuestion]:
        """Get random questions meeting criteria"""
        candidates = self.get_questions_by_criteria(exam_code, subject, difficulty)
        
        if len(candidates) < count:
            logger.warning(f"Requested {count} questions, only {len(candidates)} available")
        
        return random.sample(candidates, min(count, len(candidates)))
    
    def add_question(self, question: TestQuestion):
        """Add new question to the bank"""
        self.questions.append(question)
        key = f"{question.exam_code}_{question.subject}_{question.difficulty}"
        if key not in self.question_index:
            self.question_index[key] = []
        self.question_index[key].append(question)
    
    def update_question_quality(self, question_id: str, performance_data: Dict):
        """Update question quality metrics based on student performance"""
        for question in self.questions:
            if question.question_id == question_id:
                # Update discriminative power based on correct/incorrect rate
                correct_rate = performance_data.get('correct_rate', 0.5)
                time_stats = performance_data.get('time_stats', {})
                
                # Calculate new discriminative power
                ideal_correct_rate = 0.5  # Questions with 50% correct rate are most discriminative
                difficulty_factor = 1 - abs(correct_rate - ideal_correct_rate)
                
                question.discriminative_power = difficulty_factor
                
                # Update estimated time if available
                if time_stats and 'mean' in time_stats:
                    question.estimated_time = time_stats['mean']
                
                break

class StudentProfiler:
    """Analyzes and tracks student performance"""
    
    def __init__(self, profiles_path: str = "data/student_profiles"):
        self.profiles_path = Path(profiles_path)
        self.profiles_path.mkdir(exist_ok=True)
        self.profiles = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load existing student profiles"""
        profile_files = list(self.profiles_path.glob("*.json"))
        
        for file_path in profile_files:
            try:
                with open(file_path, 'r') as f:
                    profile_data = json.load(f)
                    student_id = file_path.stem
                    self.profiles[student_id] = StudentProfile(**profile_data)
            except Exception as e:
                logger.error(f"Error loading profile {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.profiles)} student profiles")
    
    def _save_profile(self, student_id: str, profile: StudentProfile):
        """Save student profile to file"""
        file_path = self.profiles_path / f"{student_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(asdict(profile), f, indent=2, default=str)
    
    def get_or_create_profile(self, student_id: str) -> StudentProfile:
        """Get existing profile or create new one"""
        if student_id in self.profiles:
            return self.profiles[student_id]
        
        # Create new profile
        new_profile = StudentProfile(
            student_id=student_id,
            exam_preferences=[],
            subject_strengths={},
            subject_weaknesses={},
            average_score=0.0,
            total_attempts=0,
            recent_performance=[],
            learning_curve=[],
            preferred_difficulty="medium",
            time_per_question={}
        )
        
        self.profiles[student_id] = new_profile
        self._save_profile(student_id, new_profile)
        
        return new_profile
    
    def update_profile(self, student_id: str, test_results: List[EvaluationResult]):
        """Update student profile based on test results"""
        profile = self.get_or_create_profile(student_id)
        
        # Calculate overall performance
        total_marks = sum(result.marks_awarded for result in test_results)
        total_possible = sum(result.max_marks for result in test_results)
        score_percentage = (total_marks / total_possible * 100) if total_possible > 0 else 0
        
        # Update recent performance
        profile.recent_performance.append(score_percentage)
        if len(profile.recent_performance) > 10:  # Keep only last 10 tests
            profile.recent_performance.pop(0)
        
        # Update average score
        profile.total_attempts += 1
        profile.average_score = np.mean(profile.recent_performance)
        
        # Analyze subject-wise performance
        subject_scores = defaultdict(list)
        for result in test_results:
            # Extract subject from detailed scores if available
            # This would need to be enhanced with actual subject information
            subject_scores['general'].append(result.percentage)
        
        # Update subject strengths and weaknesses
        for subject, scores in subject_scores.items():
            avg_score = np.mean(scores)
            if avg_score >= 70:
                profile.subject_strengths[subject] = avg_score
                if subject in profile.subject_weaknesses:
                    del profile.subject_weaknesses[subject]
            elif avg_score < 50:
                profile.subject_weaknesses[subject] = avg_score
                if subject in profile.subject_strengths:
                    del profile.subject_strengths[subject]
            else:
                # Remove from both if in neutral range
                profile.subject_strengths.pop(subject, None)
                profile.subject_weaknesses.pop(subject, None)
        
        # Determine preferred difficulty
        if score_percentage >= 80:
            profile.preferred_difficulty = "hard"
        elif score_percentage <= 50:
            profile.preferred_difficulty = "easy"
        else:
            profile.preferred_difficulty = "medium"
        
        # Update learning curve
        if len(profile.recent_performance) >= 3:
            # Simple trend analysis
            recent_scores = profile.recent_performance[-3:]
            if recent_scores[-1] > recent_scores[0]:
                profile.learning_curve.append("improving")
            elif recent_scores[-1] < recent_scores[0]:
                profile.learning_curve.append("declining")
            else:
                profile.learning_curve.append("stable")
        
        # Save updated profile
        self._save_profile(student_id, profile)
        
        return profile
    
    def get_personalized_recommendations(self, student_id: str) -> Dict:
        """Generate personalized study recommendations"""
        profile = self.get_or_create_profile(student_id)
        
        recommendations = {
            'focus_subjects': [],
            'difficulty_advice': '',
            'study_time_recommendation': '',
            'strengths_to_leverage': [],
            'improvement_areas': []
        }
        
        # Identify focus subjects (weaknesses)
        if profile.subject_weaknesses:
            sorted_weaknesses = sorted(profile.subject_weaknesses.items(), 
                                     key=lambda x: x[1])
            recommendations['focus_subjects'] = [subject for subject, score in sorted_weaknesses[:3]]
            recommendations['improvement_areas'] = recommendations['focus_subjects']
        
        # Identify strengths to leverage
        if profile.subject_strengths:
            sorted_strengths = sorted(profile.subject_strengths.items(), 
                                    key=lambda x: x[1], reverse=True)
            recommendations['strengths_to_leverage'] = [subject for subject, score in sorted_strengths[:2]]
        
        # Difficulty advice
        if profile.average_score < 50:
            recommendations['difficulty_advice'] = "Focus on easy and medium difficulty questions to build confidence"
        elif profile.average_score > 80:
            recommendations['difficulty_advice'] = "Challenge yourself with hard difficulty questions"
        else:
            recommendations['difficulty_advice'] = "Continue with medium difficulty questions while gradually increasing challenge"
        
        # Study time recommendation
        if len(profile.recent_performance) >= 2:
            if profile.recent_performance[-1] > profile.recent_performance[-2]:
                recommendations['study_time_recommendation'] = "Great progress! Maintain current study pace"
            else:
                recommendations['study_time_recommendation'] = "Consider increasing study time and seeking additional help"
        
        return recommendations

class AdaptiveTestGenerator:
    """Generates adaptive tests based on student performance"""
    
    def __init__(self, question_bank: QuestionBank, student_profiler: StudentProfiler):
        self.question_bank = question_bank
        self.student_profiler = student_profiler
        self.exam_config = get_exam_config()
    
    def generate_test(self, student_id: str, specification: TestSpecification) -> GeneratedTest:
        """Generate a complete test for a student"""
        test_id = f"test_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get student profile
        profile = self.student_profiler.get_or_create_profile(student_id)
        
        # Generate questions based on specification and student profile
        questions = self._select_questions_adaptively(specification, profile)
        
        # Calculate test metadata
        total_time = sum(q.estimated_time for q in questions)
        total_marks = sum(q.marks for q in questions)
        
        # Create adaptive scoring targets
        adaptive_scores = self._calculate_adaptive_targets(profile, specification)
        
        # Generate test
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=specification.duration_minutes)
        
        generated_test = GeneratedTest(
            test_id=test_id,
            student_id=student_id,
            specification=specification,
            questions=questions,
            start_time=start_time,
            end_time=end_time,
            adaptive_scores=adaptive_scores,
            metadata={
                'generated_at': start_time.isoformat(),
                'total_questions': len(questions),
                'total_time_estimated': total_time,
                'total_marks': total_marks,
                'student_profile': asdict(profile)
            }
        )
        
        logger.info(f"Generated test {test_id} with {len(questions)} questions")
        return generated_test
    
    def _select_questions_adaptively(self, spec: TestSpecification, profile: StudentProfile) -> List[TestQuestion]:
        """Select questions adaptively based on student profile"""
        selected_questions = []
        remaining_questions = spec.total_questions
        
        # Get questions by subject distribution
        for subject, percentage in spec.subject_distribution.items():
            subject_count = int(spec.total_questions * percentage / 100)
            
            # Adjust based on student strengths/weaknesses
            if subject in profile.subject_strengths:
                # Student is strong in this subject - can handle more challenging questions
                difficulty_preference = "hard" if profile.preferred_difficulty == "hard" else "medium"
            elif subject in profile.subject_weaknesses:
                # Student is weak in this subject - focus on easier questions
                difficulty_preference = "easy" if profile.preferred_difficulty == "easy" else "medium"
            else:
                difficulty_preference = profile.preferred_difficulty
            
            # Get questions for this subject
            subject_questions = self.question_bank.get_questions_by_criteria(
                exam_code=spec.exam_code,
                subject=subject,
                limit=subject_count * 2  # Get extra to select from
            )
            
            # Select questions with preferred difficulty
            selected_subject_questions = []
            for difficulty in [difficulty_preference, "medium", "easy", "hard"]:
                if len(selected_subject_questions) >= subject_count:
                    break
                
                difficulty_questions = [q for q in subject_questions if q.difficulty == difficulty]
                needed = subject_count - len(selected_subject_questions)
                selected_subject_questions.extend(difficulty_questions[:needed])
            
            selected_questions.extend(selected_subject_questions[:subject_count])
            remaining_questions -= len(selected_subject_questions[:subject_count])
        
        # Fill remaining questions randomly
        if remaining_questions > 0:
            remaining_candidates = self.question_bank.get_questions_by_criteria(
                exam_code=spec.exam_code,
                limit=remaining_questions * 2
            )
            
            # Filter out already selected questions
            selected_ids = {q.question_id for q in selected_questions}
            remaining_candidates = [q for q in remaining_candidates if q.question_id not in selected_ids]
            
            # Fill with random selection
            additional_questions = random.sample(
                remaining_candidates, 
                min(remaining_questions, len(remaining_candidates))
            )
            selected_questions.extend(additional_questions)
        
        # Shuffle questions to mix difficulty and subjects
        random.shuffle(selected_questions)
        
        return selected_questions[:spec.total_questions]
    
    def _calculate_adaptive_targets(self, profile: StudentProfile, spec: TestSpecification) -> Dict[str, float]:
        """Calculate adaptive scoring targets for the test"""
        targets = {}
        
        # Overall target based on student performance
        if profile.average_score > 0:
            targets['overall'] = min(profile.average_score + 5, 95.0)  # Slight improvement target
        else:
            targets['overall'] = 70.0  # Default target for new students
        
        # Subject-wise targets
        for subject in spec.subject_distribution.keys():
            if subject in profile.subject_strengths:
                targets[subject] = min(profile.subject_strengths[subject] + 3, 95.0)
            elif subject in profile.subject_weaknesses:
                targets[subject] = max(profile.subject_weaknesses[subject] + 10, 50.0)
            else:
                targets[subject] = targets['overall']
        
        return targets
    
    def generate_standard_test(self, exam_code: str, total_questions: int = 100, 
                             duration: int = 120) -> GeneratedTest:
        """Generate a standard test following official exam pattern"""
        # Get exam configuration
        exam_config = None
        for category_exams in self.exam_config.values():
            if exam_code in category_exams:
                exam_config = category_exams[exam_code]
                break
        
        if not exam_config:
            raise ValueError(f"Exam configuration not found for {exam_code}")
        
        # Create standard specification
        subject_distribution = {}
        total_subjects = len(exam_config['subjects'])
        equal_distribution = 100.0 / total_subjects
        
        for subject in exam_config['subjects']:
            subject_distribution[subject] = equal_distribution
        
        # Difficulty distribution based on exam type
        if exam_code in ['upsc_cse', 'rbi_gradeb']:
            difficulty_distribution = {'easy': 20, 'medium': 50, 'hard': 30}
        else:
            difficulty_distribution = {'easy': 40, 'medium': 50, 'hard': 10}
        
        specification = TestSpecification(
            exam_code=exam_code,
            total_questions=total_questions,
            total_marks=total_questions,  # Assuming 1 mark per question
            duration_minutes=duration,
            subject_distribution=subject_distribution,
            difficulty_distribution=difficulty_distribution,
            include_explanations=True,
            adaptive=False
        )
        
        # Generate test without student profile (standard)
        test_id = f"standard_{exam_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Select questions according to specifications
        selected_questions = []
        for subject, percentage in subject_distribution.items():
            subject_count = int(total_questions * percentage / 100)
            
            # Distribute by difficulty
            for difficulty, difficulty_percentage in difficulty_distribution.items():
                difficulty_count = int(subject_count * difficulty_percentage / 100)
                
                questions = self.question_bank.get_questions_by_criteria(
                    exam_code=exam_code,
                    subject=subject,
                    difficulty=difficulty,
                    limit=difficulty_count
                )
                
                selected_questions.extend(questions)
        
        # Fill any remaining spots
        remaining = total_questions - len(selected_questions)
        if remaining > 0:
            remaining_questions = self.question_bank.get_questions_by_criteria(
                exam_code=exam_code,
                limit=remaining
            )
            selected_questions.extend(remaining_questions[:remaining])
        
        # Shuffle and limit to required count
        random.shuffle(selected_questions)
        selected_questions = selected_questions[:total_questions]
        
        # Calculate total marks and time
        total_marks = sum(q.marks for q in selected_questions)
        total_time = sum(q.estimated_time for q in selected_questions)
        
        generated_test = GeneratedTest(
            test_id=test_id,
            student_id="standard",
            specification=specification,
            questions=selected_questions,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=duration),
            adaptive_scores={},
            metadata={
                'generated_at': datetime.now().isoformat(),
                'test_type': 'standard',
                'total_marks': total_marks,
                'estimated_time': total_time
            }
        )
        
        return generated_test
    
    def save_test(self, test: GeneratedTest, output_dir: str = "tests/generated"):
        """Save generated test to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert to serializable format
        test_data = {
            'test_id': test.test_id,
            'student_id': test.student_id,
            'specification': asdict(test.specification),
            'questions': [asdict(q) for q in test.questions],
            'start_time': test.start_time.isoformat(),
            'end_time': test.end_time.isoformat(),
            'adaptive_scores': test.adaptive_scores,
            'metadata': test.metadata
        }
        
        file_path = output_path / f"{test.test_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved test {test.test_id} to {file_path}")
        return str(file_path)

# Usage example
if __name__ == "__main__":
    # Initialize components
    question_bank = QuestionBank()
    student_profiler = StudentProfiler()
    test_generator = AdaptiveTestGenerator(question_bank, student_profiler)
    
    # Create test specification
    spec = TestSpecification(
        exam_code="ssc_cgl",
        total_questions=50,
        total_marks=50,
        duration_minutes=60,
        subject_distribution={
            'english': 25,
            'quantitative_aptitude': 25,
            'reasoning': 25,
            'general_awareness': 25
        },
        difficulty_distribution={'easy': 40, 'medium': 50, 'hard': 10},
        include_explanations=True,
        adaptive=True
    )
    
    # Generate adaptive test for a student
    student_id = "student_001"
    test = test_generator.generate_test(student_id, spec)
    
    # Save test
    test_file = test_generator.save_test(test)
    print(f"Generated adaptive test: {test.test_id}")
    print(f"Questions: {len(test.questions)}")
    print(f"Total marks: {sum(q.marks for q in test.questions)}")
    print(f"Saved to: {test_file}")
    
    # Generate standard test
    standard_test = test_generator.generate_standard_test("ssc_cgl", 100, 120)
    standard_file = test_generator.save_test(standard_test)
    print(f"Generated standard test: {standard_test.test_id}")
    print(f"Saved to: {standard_file}")
    
    # Get personalized recommendations
    recommendations = student_profiler.get_personalized_recommendations(student_id)
    print(f"Personalized recommendations: {recommendations}")