"""
Answer Evaluation Engine for Government Exams
NLP-based system for automatic evaluation of descriptive answers
Supports both objective (MCQ) and subjective answer evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import re
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
import warnings

# NLP and ML libraries
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import transformers
from transformers import pipeline, AutoTokenizer, AutoModel

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class Answer:
    """Answer data structure"""
    question_id: str
    student_answer: str
    correct_answer: str
    answer_type: str  # 'objective', 'subjective', 'essay'
    max_marks: float
    subject: Optional[str] = None
    topic: Optional[str] = None
    keywords: Optional[List[str]] = None

@dataclass
class EvaluationResult:
    """Result of answer evaluation"""
    question_id: str
    marks_awarded: float
    max_marks: float
    percentage: float
    detailed_scores: Dict[str, float]
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    keyword_matches: List[str]
    missing_keywords: List[str]
    quality_score: float

class TextProcessor:
    """Text processing utilities for answer evaluation"""
    
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_available = True
        except Exception:
            logger.warning("Sentence transformers not available, using TF-IDF similarity")
            self.semantic_available = False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(self.preprocess_text(text))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalpha()]
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text using TF-IDF"""
        # Simple keyword extraction based on frequency and length
        tokens = self.tokenize_and_lemmatize(text)
        
        # Count word frequencies
        word_freq = {}
        for word in tokens:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.semantic_available:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except Exception:
                pass
        
        # Fallback to TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

class KeywordMatcher:
    """Matches keywords between student and correct answers"""
    
    def __init__(self):
        self.synonym_dict = self._load_synonyms()
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary for keyword matching"""
        return {
            'government': ['state', 'public', 'official', 'administrative'],
            'political': ['governmental', 'civic', 'public', 'democratic'],
            'economic': ['financial', 'monetary', 'fiscal', 'trade'],
            'social': ['societal', 'community', 'public', 'civil'],
            'constitution': ['fundamental', 'basic', 'constitutional', 'charter'],
            'development': ['growth', 'progress', 'advancement', 'improvement'],
            'policy': ['strategy', 'plan', 'program', 'initiative'],
            'system': ['framework', 'structure', 'mechanism', 'process'],
            'important': ['significant', 'crucial', 'vital', 'key'],
            'main': ['primary', 'principal', 'major', 'chief']
        }
    
    def find_keyword_matches(self, student_answer: str, correct_answer: str, keywords: List[str]) -> Tuple[List[str], List[str]]:
        """Find matching and missing keywords"""
        student_tokens = set(self._get_tokens(student_answer))
        correct_tokens = set(self._get_tokens(correct_answer))
        
        matched_keywords = []
        missing_keywords = []
        
        for keyword in keywords:
            keyword_tokens = set(self._get_tokens(keyword))
            
            # Check for exact matches
            if keyword_tokens.intersection(student_tokens):
                matched_keywords.append(keyword)
            else:
                # Check for synonym matches
                synonym_matches = False
                for token in keyword_tokens:
                    if token in self.synonym_dict:
                        synonym_tokens = set(self.synonym_dict[token])
                        if synonym_tokens.intersection(student_tokens):
                            matched_keywords.append(f"{keyword} (synonym)")
                            synonym_matches = True
                            break
                
                if not synonym_matches:
                    missing_keywords.append(keyword)
        
        return matched_keywords, missing_keywords
    
    def _get_tokens(self, text: str) -> List[str]:
        """Extract tokens from text"""
        return re.findall(r'\b\w+\b', text.lower())

class ObjectiveAnswerEvaluator:
    """Evaluates multiple choice and objective answers"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def evaluate_mcq(self, student_answer: str, correct_answer: str, options: List[str] = None) -> EvaluationResult:
        """Evaluate multiple choice question"""
        student_answer = student_answer.strip().upper()
        correct_answer = correct_answer.strip().upper()
        
        is_correct = student_answer == correct_answer
        
        # Check for similar answers (A vs a, etc.)
        if not is_correct and student_answer and correct_answer:
            is_correct = student_answer[0] == correct_answer[0]  # Handle A) vs A
        
        marks_awarded = 1.0 if is_correct else 0.0
        percentage = 100.0 if is_correct else 0.0
        
        feedback = "Correct answer!" if is_correct else f"Incorrect. The correct answer is {correct_answer}."
        
        return EvaluationResult(
            question_id="",
            marks_awarded=marks_awarded,
            max_marks=1.0,
            percentage=percentage,
            detailed_scores={'accuracy': 1.0 if is_correct else 0.0},
            feedback=feedback,
            strengths=["Correct selection" if is_correct else "Attempted the question"],
            weaknesses=[] if is_correct else ["Incorrect answer choice"],
            keyword_matches=[],
            missing_keywords=[],
            quality_score=1.0 if is_correct else 0.0
        )

class SubjectiveAnswerEvaluator:
    """Evaluates descriptive and essay answers"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.keyword_matcher = KeywordMatcher()
        
        # Initialize ROUGE scorer for summary evaluation
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_available = True
        except Exception:
            logger.warning("ROUGE scorer not available")
            self.rouge_available = False
        
        # Initialize BLEU scorer
        self.smoothing = SmoothingFunction().method1
        
        # Set weights for different scoring criteria
        self.scoring_weights = {
            'keyword_match': 0.3,
            'semantic_similarity': 0.25,
            'content_coverage': 0.2,
            'structural_quality': 0.15,
            'language_quality': 0.1
        }
    
    def evaluate_descriptive(self, answer: Answer) -> EvaluationResult:
        """Evaluate descriptive answer"""
        student_tokens = self.text_processor.tokenize_and_lemmatize(answer.student_answer)
        correct_tokens = self.text_processor.tokenize_and_lemmatize(answer.correct_answer)
        
        # Calculate different similarity scores
        keyword_score = self._calculate_keyword_score(answer)
        semantic_score = self._calculate_semantic_score(answer)
        coverage_score = self._calculate_content_coverage(student_tokens, correct_tokens)
        structure_score = self._calculate_structural_quality(answer.student_answer)
        language_score = self._calculate_language_quality(answer.student_answer)
        
        # Calculate weighted final score
        final_score = (
            keyword_score * self.scoring_weights['keyword_match'] +
            semantic_score * self.scoring_weights['semantic_similarity'] +
            coverage_score * self.scoring_weights['content_coverage'] +
            structure_score * self.scoring_weights['structural_quality'] +
            language_score * self.scoring_weights['language_quality']
        )
        
        # Convert to marks
        marks_awarded = final_score * answer.max_marks
        percentage = final_score * 100
        
        # Generate detailed feedback
        feedback = self._generate_feedback(final_score, keyword_score, semantic_score, coverage_score)
        strengths, weaknesses = self._analyze_performance(keyword_score, semantic_score, coverage_score)
        
        # Keyword analysis
        matched_keywords, missing_keywords = self.keyword_matcher.find_keyword_matches(
            answer.student_answer, answer.correct_answer, answer.keywords or []
        )
        
        return EvaluationResult(
            question_id=answer.question_id,
            marks_awarded=marks_awarded,
            max_marks=answer.max_marks,
            percentage=percentage,
            detailed_scores={
                'keyword_match': keyword_score,
                'semantic_similarity': semantic_score,
                'content_coverage': coverage_score,
                'structural_quality': structure_score,
                'language_quality': language_score,
                'overall_score': final_score
            },
            feedback=feedback,
            strengths=strengths,
            weaknesses=weaknesses,
            keyword_matches=matched_keywords,
            missing_keywords=missing_keywords,
            quality_score=final_score
        )
    
    def _calculate_keyword_score(self, answer: Answer) -> float:
        """Calculate keyword matching score"""
        if not answer.keywords:
            return 0.5  # Neutral score if no keywords provided
        
        matched_keywords, missing_keywords = self.keyword_matcher.find_keyword_matches(
            answer.student_answer, answer.correct_answer, answer.keywords
        )
        
        if len(answer.keywords) == 0:
            return 0.5
        
        score = len(matched_keywords) / len(answer.keywords)
        return min(score, 1.0)
    
    def _calculate_semantic_score(self, answer: Answer) -> float:
        """Calculate semantic similarity score"""
        similarity = self.text_processor.semantic_similarity(
            answer.student_answer, answer.correct_answer
        )
        return similarity
    
    def _calculate_content_coverage(self, student_tokens: List[str], correct_tokens: List[str]) -> float:
        """Calculate content coverage using Jaccard similarity"""
        if not student_tokens or not correct_tokens:
            return 0.0
        
        intersection = len(set(student_tokens).intersection(set(correct_tokens)))
        union = len(set(student_tokens).union(set(correct_tokens)))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_structural_quality(self, student_answer: str) -> float:
        """Calculate structural quality score"""
        sentences = sent_tokenize(student_answer)
        
        if not sentences:
            return 0.0
        
        # Factors: sentence count, average sentence length, paragraph structure
        score = 0.0
        
        # Sentence count (ideal: 3-10 sentences)
        sentence_count = len(sentences)
        if 3 <= sentence_count <= 10:
            score += 0.3
        elif sentence_count > 10:
            score += 0.2  # Penalize overly long answers
        
        # Average sentence length (ideal: 10-20 words)
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
        if 10 <= avg_sentence_length <= 20:
            score += 0.3
        elif avg_sentence_length > 20:
            score += 0.2
        
        # Paragraph breaks (simple heuristic)
        if '\n' in student_answer or '\n\n' in student_answer:
            score += 0.2
        
        # Introduction and conclusion indicators
        intro_words = ['introduction', 'begin', 'start', 'initially', 'firstly']
        conclusion_words = ['conclusion', 'finally', 'in conclusion', 'to conclude', 'summarize']
        
        answer_lower = student_answer.lower()
        if any(word in answer_lower for word in intro_words):
            score += 0.1
        if any(word in answer_lower for word in conclusion_words):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_language_quality(self, student_answer: str) -> float:
        """Calculate language quality score"""
        # Simple heuristics for language quality
        score = 0.0
        
        # Sentence structure (no excessive repetition)
        words = word_tokenize(student_answer.lower())
        if words:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            score += min(repetition_ratio, 0.5)  # Higher variety = better score
        
        # Grammar indicators (very basic)
        grammar_indicators = ['because', 'therefore', 'however', 'although', 'since', 'thus']
        answer_lower = student_answer.lower()
        grammar_score = sum(1 for indicator in grammar_indicators if indicator in answer_lower)
        score += min(grammar_score * 0.1, 0.3)
        
        # Length appropriateness (not too short, not too long)
        word_count = len(words)
        if 50 <= word_count <= 300:  # Reasonable length for descriptive answer
            score += 0.2
        elif word_count < 50:
            score += 0.1  # Too short
        # Penalize overly long answers
        elif word_count > 500:
            score -= 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _generate_feedback(self, overall_score: float, keyword_score: float, 
                          semantic_score: float, coverage_score: float) -> str:
        """Generate detailed feedback based on scores"""
        if overall_score >= 0.8:
            feedback = "Excellent answer! "
        elif overall_score >= 0.6:
            feedback = "Good answer. "
        elif overall_score >= 0.4:
            feedback = "Average answer. "
        else:
            feedback = "Needs improvement. "
        
        # Specific feedback based on scores
        if keyword_score < 0.5:
            feedback += "Include more key concepts and terminology. "
        
        if semantic_score < 0.5:
            feedback += "Improve content alignment with the correct answer. "
        
        if coverage_score < 0.5:
            feedback += "Cover more aspects of the topic comprehensively. "
        
        return feedback
    
    def _analyze_performance(self, keyword_score: float, semantic_score: float, 
                           coverage_score: float) -> Tuple[List[str], List[str]]:
        """Analyze performance and generate strengths/weaknesses"""
        strengths = []
        weaknesses = []
        
        if keyword_score >= 0.7:
            strengths.append("Good use of key terminology")
        else:
            weaknesses.append("Missing important keywords")
        
        if semantic_score >= 0.6:
            strengths.append("Content well-aligned with requirements")
        else:
            weaknesses.append("Content not well-aligned with expected answer")
        
        if coverage_score >= 0.6:
            strengths.append("Comprehensive coverage of topic")
        else:
            weaknesses.append("Limited topic coverage")
        
        return strengths, weaknesses

class AnswerEvaluationEngine:
    """Main answer evaluation engine"""
    
    def __init__(self):
        self.objective_evaluator = ObjectiveAnswerEvaluator()
        self.subjective_evaluator = SubjectiveAnswerEvaluator()
        self.text_processor = TextProcessor()
    
    def evaluate_answer(self, answer: Answer) -> EvaluationResult:
        """Evaluate a single answer"""
        if answer.answer_type == 'objective':
            if len(answer.student_answer) <= 2:  # Likely MCQ answer
                return self.objective_evaluator.evaluate_mcq(
                    answer.student_answer, answer.correct_answer, answer.keywords
                )
        
        # For descriptive answers
        return self.subjective_evaluator.evaluate_descriptive(answer)
    
    def batch_evaluate(self, answers: List[Answer]) -> List[EvaluationResult]:
        """Evaluate multiple answers"""
        results = []
        for answer in answers:
            try:
                result = self.evaluate_answer(answer)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating answer {answer.question_id}: {str(e)}")
                # Create error result
                results.append(EvaluationResult(
                    question_id=answer.question_id,
                    marks_awarded=0.0,
                    max_marks=answer.max_marks,
                    percentage=0.0,
                    detailed_scores={},
                    feedback=f"Evaluation error: {str(e)}",
                    strengths=[],
                    weaknesses=["Evaluation failed"],
                    keyword_matches=[],
                    missing_keywords=[],
                    quality_score=0.0
                ))
        
        return results
    
    def get_detailed_report(self, answers: List[Answer], results: List[EvaluationResult]) -> Dict:
        """Generate detailed evaluation report"""
        total_marks = sum(result.marks_awarded for result in results)
        total_max_marks = sum(result.max_marks for result in results)
        overall_percentage = (total_marks / total_max_marks * 100) if total_max_marks > 0 else 0
        
        # Analyze subject-wise performance
        subject_performance = {}
        for i, (answer, result) in enumerate(zip(answers, results)):
            if answer.subject:
                if answer.subject not in subject_performance:
                    subject_performance[answer.subject] = {'obtained': 0, 'total': 0}
                subject_performance[answer.subject]['obtained'] += result.marks_awarded
                subject_performance[answer.subject]['total'] += result.max_marks
        
        # Calculate subject-wise percentages
        for subject in subject_performance:
            subject_data = subject_performance[subject]
            subject_performance[subject]['percentage'] = (
                subject_data['obtained'] / subject_data['total'] * 100
            )
        
        report = {
            'overall': {
                'total_marks': total_marks,
                'max_marks': total_max_marks,
                'percentage': overall_percentage,
                'grade': self._get_grade(overall_percentage)
            },
            'subject_performance': subject_performance,
            'detailed_results': [
                {
                    'question_id': result.question_id,
                    'marks': result.marks_awarded,
                    'percentage': result.percentage,
                    'feedback': result.feedback,
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses
                }
                for result in results
            ]
        }
        
        return report
    
    def _get_grade(self, percentage: float) -> str:
        """Convert percentage to grade"""
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B+"
        elif percentage >= 60:
            return "B"
        elif percentage >= 50:
            return "C"
        else:
            return "F"

# Usage example and testing
if __name__ == "__main__":
    # Sample answers for testing
    test_answers = [
        Answer(
            question_id="Q1",
            student_answer="Paris",
            correct_answer="C",
            answer_type="objective",
            max_marks=1.0,
            subject="geography"
        ),
        Answer(
            question_id="Q2",
            student_answer="The constitution is the supreme law of the land that establishes the fundamental political principles, structures of government, and defines the rights and duties of citizens.",
            correct_answer="The constitution serves as the supreme legal document that establishes the fundamental political framework, government structure, and defines the basic rights and obligations of citizens in a democratic society.",
            answer_type="subjective",
            max_marks=5.0,
            subject="polity",
            keywords=['constitution', 'supreme', 'fundamental', 'government', 'rights', 'citizens']
        )
    ]
    
    # Initialize evaluation engine
    evaluator = AnswerEvaluationEngine()
    
    # Evaluate answers
    results = evaluator.batch_evaluate(test_answers)
    
    # Display results
    for answer, result in zip(test_answers, results):
        print(f"Question: {answer.question_id}")
        print(f"Marks: {result.marks_awarded}/{result.max_marks} ({result.percentage:.1f}%)")
        print(f"Feedback: {result.feedback}")
        print(f"Strengths: {', '.join(result.strengths)}")
        print(f"Weaknesses: {', '.join(result.weaknesses)}")
        print("-" * 50)
    
    # Generate detailed report
    report = evaluator.get_detailed_report(test_answers, results)
    print(f"\nOverall Performance:")
    print(f"Total Marks: {report['overall']['total_marks']}/{report['overall']['max_marks']}")
    print(f"Percentage: {report['overall']['percentage']:.1f}%")
    print(f"Grade: {report['overall']['grade']}")