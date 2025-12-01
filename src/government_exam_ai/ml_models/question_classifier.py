"""
Question Classification System for Government Exams
Uses ML models to classify questions by subject, topic, difficulty, and other attributes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import json
from pathlib import Path
import logging
from dataclasses import dataclass
import re
from collections import Counter, defaultdict

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

logger = logging.getLogger(__name__)

@dataclass
class Question:
    """Question data structure"""
    text: str
    options: List[str]
    correct_answer: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    exam_code: Optional[str] = None
    year: Optional[int] = None

@dataclass
class ClassificationResult:
    """Result of question classification"""
    question: Question
    predicted_subject: str
    subject_confidence: float
    predicted_topic: str
    topic_confidence: float
    predicted_difficulty: str
    difficulty_confidence: float
    subject_probabilities: Dict[str, float]
    topic_probabilities: Dict[str, float]
    difficulty_probabilities: Dict[str, float]

class TextPreprocessor:
    """Text preprocessing for question classification"""
    
    def __init__(self):
        # Download required NLTK data
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
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\?\!\:]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.nlp:
            # Use spaCy for better tokenization
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        else:
            # Fallback to NLTK
            return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_features(self, question: Question) -> Dict:
        """Extract features from question for classification"""
        text_features = self.extract_text_features(question)
        option_features = self.extract_option_features(question)
        structural_features = self.extract_structural_features(question)
        
        return {**text_features, **option_features, **structural_features}
    
    def extract_text_features(self, question: Question) -> Dict:
        """Extract features from question text"""
        clean_text = self.clean_text(question.text)
        tokens = self.tokenize(clean_text)
        tokens_no_stop = self.remove_stop_words(tokens)
        
        # Basic text statistics
        features = {
            'question_length': len(clean_text),
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
            'question_marks': question.text.count('?'),
            'numbers_count': len(re.findall(r'\d+', clean_text)),
            'capital_ratio': sum(1 for c in question.text if c.isupper()) / len(question.text) if question.text else 0
        }
        
        # Subject-specific keywords
        subject_keywords = {
            'mathematics': ['calculate', 'value', 'sum', 'product', 'equation', 'solve', 'formula', 'number'],
            'english': ['meaning', 'synonym', 'antonym', 'passage', 'grammar', 'comprehension', 'word'],
            'reasoning': ['logical', 'sequence', 'pattern', 'arrangement', 'classification', 'analogy'],
            'history': ['ancient', 'medieval', 'modern', 'century', 'year', 'war', 'empire', 'king'],
            'geography': ['river', 'mountain', 'country', 'capital', 'latitude', 'longitude', 'climate'],
            'polity': ['constitution', 'parliament', 'president', 'minister', 'election', 'law', 'amendment'],
            'economy': ['gdp', 'inflation', 'bank', 'currency', 'trade', 'budget', 'fiscal', 'monetary'],
            'science': ['element', 'atom', 'molecule', 'reaction', 'formula', 'experiment', 'theory']
        }
        
        # Count keyword matches
        text_lower = clean_text.lower()
        for subject, keywords in subject_keywords.items():
            features[f'{subject}_keywords'] = sum(1 for keyword in keywords if keyword in text_lower)
        
        return features
    
    def extract_option_features(self, question: Question) -> Dict:
        """Extract features from answer options"""
        if not question.options:
            return {'option_count': 0}
        
        options_text = ' '.join(question.options)
        features = {
            'option_count': len(question.options),
            'avg_option_length': np.mean([len(opt) for opt in question.options]),
            'option_numbers': sum(1 for opt in question.options if any(char.isdigit() for char in opt)),
            'option_years': len(re.findall(r'\b(19|20)\d{2}\b', options_text))
        }
        
        return features
    
    def extract_structural_features(self, question: Question) -> Dict:
        """Extract structural features from question"""
        features = {
            'has_if_clause': 1 if 'if' in question.text.lower() else 0,
            'has_which_clause': 1 if 'which' in question.text.lower() else 0,
            'has_what_clause': 1 if 'what' in question.text.lower() else 0,
            'has_when_clause': 1 if 'when' in question.text.lower() else 0,
            'has_where_clause': 1 if 'where' in question.text.lower() else 0,
            'has_who_clause': 1 if 'who' in question.text.lower() else 0,
            'has_how_clause': 1 if 'how' in question.text.lower() else 0,
            'has_why_clause': 1 if 'why' in question.text.lower() else 0
        }
        
        return features

class QuestionClassifier:
    """Multi-class question classification system"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
        self.subject_encoder = LabelEncoder()
        self.topic_encoder = LabelEncoder()
        self.difficulty_encoder = LabelEncoder()
        
        # ML models for different classification tasks
        self.subject_classifier = None
        self.topic_classifier = None
        self.difficulty_classifier = None
        
        # Vectorizers for text features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.count_vectorizer = CountVectorizer(max_features=500)
        
        self.is_trained = False
    
    def prepare_features(self, questions: List[Question]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        logger.info("Preparing features for classification...")
        
        # Extract text features
        texts = [q.text for q in questions]
        
        # Create TF-IDF vectors
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        
        # Extract additional features
        additional_features = []
        for question in questions:
            features = self.preprocessor.extract_features(question)
            additional_features.append(list(features.values()))
        
        additional_features = np.array(additional_features)
        
        # Combine features
        X = np.hstack([tfidf_features, additional_features])
        
        # Prepare labels
        subjects = [q.subject for q in questions if q.subject]
        topics = [q.topic for q in questions if q.topic]
        difficulties = [q.difficulty for q in questions if q.difficulty]
        
        y_subject = self.subject_encoder.fit_transform(subjects)
        y_topic = self.topic_encoder.fit_transform(topics)
        y_difficulty = self.difficulty_encoder.fit_transform(difficultities)
        
        return X, y_subject, y_topic, y_difficulty
    
    def train(self, questions: List[Question]) -> Dict:
        """Train classification models"""
        logger.info("Training question classification models...")
        
        if len(questions) < 10:
            raise ValueError("Need at least 10 questions to train classification models")
        
        # Prepare data
        X, y_subject, y_topic, y_difficulty = self.prepare_features(questions)
        
        # Train subject classifier
        logger.info("Training subject classifier...")
        self.subject_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.subject_classifier.fit(X, y_subject)
        
        # Train topic classifier
        logger.info("Training topic classifier...")
        self.topic_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.topic_classifier.fit(X, y_topic)
        
        # Train difficulty classifier
        logger.info("Training difficulty classifier...")
        self.difficulty_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.difficulty_classifier.fit(X, y_difficulty)
        
        self.is_trained = True
        
        # Evaluate models
        evaluation_results = self._evaluate_models(X, y_subject, y_topic, y_difficulty)
        
        # Save models
        self.save_models()
        
        logger.info("Training completed successfully!")
        return evaluation_results
    
    def _evaluate_models(self, X: np.ndarray, y_subject: np.ndarray, y_topic: np.ndarray, y_difficulty: np.ndarray) -> Dict:
        """Evaluate trained models"""
        logger.info("Evaluating model performance...")
        
        results = {}
        
        # Split data for evaluation
        X_train, X_test, y_sub_train, y_sub_test = train_test_split(X, y_subject, test_size=0.2, random_state=42)
        _, _, y_topic_train, y_topic_test = train_test_split(X, y_topic, test_size=0.2, random_state=42)
        _, _, y_diff_train, y_diff_test = train_test_split(X, y_difficulty, test_size=0.2, random_state=42)
        
        # Evaluate subject classifier
        subject_predictions = self.subject_classifier.predict(X_test)
        results['subject_accuracy'] = accuracy_score(y_sub_test, subject_predictions)
        results['subject_report'] = classification_report(y_sub_test, subject_predictions, target_names=self.subject_encoder.classes_)
        
        # Evaluate topic classifier
        topic_predictions = self.topic_classifier.predict(X_test)
        results['topic_accuracy'] = accuracy_score(y_topic_test, topic_predictions)
        results['topic_report'] = classification_report(y_topic_test, topic_predictions, target_names=self.topic_encoder.classes_)
        
        # Evaluate difficulty classifier
        difficulty_predictions = self.difficulty_classifier.predict(X_test)
        results['difficulty_accuracy'] = accuracy_score(y_diff_test, difficulty_predictions)
        results['difficulty_report'] = classification_report(y_diff_test, difficulty_predictions, target_names=self.difficulty_encoder.classes_)
        
        return results
    
    def predict(self, question: Question) -> ClassificationResult:
        """Classify a single question"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        tfidf_features = self.tfidf_vectorizer.transform([question.text]).toarray()
        additional_features = np.array([list(self.preprocessor.extract_features(question).values())])
        X = np.hstack([tfidf_features, additional_features])
        
        # Make predictions
        subject_pred = self.subject_classifier.predict(X)[0]
        subject_proba = self.subject_classifier.predict_proba(X)[0]
        
        topic_pred = self.topic_classifier.predict(X)[0]
        topic_proba = self.topic_classifier.predict_proba(X)[0]
        
        difficulty_pred = self.difficulty_classifier.predict(X)[0]
        difficulty_proba = self.difficulty_classifier.predict_proba(X)[0]
        
        # Decode predictions
        predicted_subject = self.subject_encoder.inverse_transform([subject_pred])[0]
        predicted_topic = self.topic_encoder.inverse_transform([topic_pred])[0]
        predicted_difficulty = self.difficulty_encoder.inverse_transform([difficulty_pred])[0]
        
        # Get confidence scores
        subject_confidence = np.max(subject_proba)
        topic_confidence = np.max(topic_proba)
        difficulty_confidence = np.max(difficulty_proba)
        
        # Create probability dictionaries
        subject_probabilities = {
            self.subject_encoder.classes_[i]: prob 
            for i, prob in enumerate(subject_proba)
        }
        topic_probabilities = {
            self.topic_encoder.classes_[i]: prob 
            for i, prob in enumerate(topic_proba)
        }
        difficulty_probabilities = {
            self.difficulty_encoder.classes_[i]: prob 
            for i, prob in enumerate(difficulty_proba)
        }
        
        return ClassificationResult(
            question=question,
            predicted_subject=predicted_subject,
            subject_confidence=subject_confidence,
            predicted_topic=predicted_topic,
            topic_confidence=topic_confidence,
            predicted_difficulty=predicted_difficulty,
            difficulty_confidence=difficulty_confidence,
            subject_probabilities=subject_probabilities,
            topic_probabilities=topic_probabilities,
            difficulty_probabilities=difficulty_probabilities
        )
    
    def batch_predict(self, questions: List[Question]) -> List[ClassificationResult]:
        """Classify multiple questions"""
        return [self.predict(question) for question in questions]
    
    def save_models(self):
        """Save trained models and encoders"""
        model_data = {
            'subject_classifier': self.subject_classifier,
            'topic_classifier': self.topic_classifier,
            'difficulty_classifier': self.difficulty_classifier,
            'subject_encoder': self.subject_encoder,
            'topic_encoder': self.topic_encoder,
            'difficulty_encoder': self.difficulty_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer
        }
        
        with open(self.model_dir / 'question_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Models saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        model_path = self.model_dir / 'question_classifier.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.subject_classifier = model_data['subject_classifier']
        self.topic_classifier = model_data['topic_classifier']
        self.difficulty_classifier = model_data['difficulty_classifier']
        self.subject_encoder = model_data['subject_encoder']
        self.topic_encoder = model_data['topic_encoder']
        self.difficulty_encoder = model_data['difficulty_encoder']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.count_vectorizer = model_data['count_vectorizer']
        
        self.is_trained = True
        logger.info("Models loaded successfully!")

class SubjectKnowledgeBase:
    """Knowledge base for subject-specific information"""
    
    def __init__(self):
        self.subject_topics = {
            'english': [
                'grammar', 'vocabulary', 'reading_comprehension', 'sentence_completion',
                'synonyms_antonyms', 'idioms_phrases', 'error_detection', 'sentence_rearrangement'
            ],
            'mathematics': [
                'arithmetic', 'algebra', 'geometry', 'trigonometry', 'calculus', 
                'statistics', 'probability', 'number_system', 'profit_loss', 'percentage'
            ],
            'reasoning': [
                'logical_reasoning', 'analytical_reasoning', 'non_verbal_reasoning',
                'sequence', 'classification', 'analogy', 'coding_decoding', 'blood_relations'
            ],
            'general_knowledge': [
                'history', 'geography', 'politics', 'economy', 'science', 'sports',
                'awards', 'books_authors', 'important_days', 'indian_constitution'
            ],
            'general_awareness': [
                'current_affairs', 'static_gk', 'economics', 'science_technology',
                'environment', 'art_culture', 'national_international_events'
            ],
            'computer_knowledge': [
                'computer_fundamentals', 'internet', 'ms_office', 'database',
                'networking', 'programming', 'cyber_security'
            ]
        }
        
        self.difficulty_patterns = {
            'easy': {
                'max_length': 50,
                'max_options': 4,
                'keywords': ['basic', 'simple', 'elementary', 'general']
            },
            'medium': {
                'max_length': 100,
                'max_options': 5,
                'keywords': ['calculate', 'determine', 'find', 'solve']
            },
            'hard': {
                'min_length': 80,
                'min_options': 4,
                'keywords': ['analyze', 'evaluate', 'synthesize', 'derive']
            }
        }
    
    def suggest_topics(self, subject: str) -> List[str]:
        """Suggest topics for a given subject"""
        return self.subject_topics.get(subject.lower(), [])
    
    def classify_difficulty_by_patterns(self, question: Question) -> str:
        """Classify difficulty based on question patterns"""
        text = question.text.lower()
        length = len(question.text)
        option_count = len(question.options)
        
        # Easy questions
        if any(keyword in text for keyword in self.difficulty_patterns['easy']['keywords']):
            if length <= 50 and option_count <= 4:
                return 'easy'
        
        # Hard questions
        if any(keyword in text for keyword in self.difficulty_patterns['hard']['keywords']):
            if length >= 80:
                return 'hard'
        
        # Default to medium
        return 'medium'

# Usage example and testing
if __name__ == "__main__":
    # Sample questions for testing
    sample_questions = [
        Question(
            text="What is the capital of France?",
            options=["London", "Berlin", "Paris", "Madrid"],
            correct_answer="C",
            subject="geography",
            topic="countries_capitals",
            difficulty="easy"
        ),
        Question(
            text="Calculate the area of a circle with radius 7 cm. (Use π = 22/7)",
            options=["154 sq cm", "147 sq cm", "140 sq cm", "161 sq cm"],
            correct_answer="A",
            subject="mathematics",
            topic="geometry",
            difficulty="medium"
        ),
        Question(
            text="Choose the word most nearly opposite in meaning to 'obstinate'",
            options=["stubborn", "flexible", "persistent", "determined"],
            correct_answer="B",
            subject="english",
            topic="synonyms_antonyms",
            difficulty="medium"
        )
    ]
    
    # Initialize and train classifier
    classifier = QuestionClassifier()
    try:
        results = classifier.train(sample_questions)
        print("Training Results:")
        print(f"Subject Accuracy: {results['subject_accuracy']:.3f}")
        print(f"Topic Accuracy: {results['topic_accuracy']:.3f}")
        print(f"Difficulty Accuracy: {results['difficulty_accuracy']:.3f}")
        
        # Test classification
        test_question = Question(
            text="What is the chemical symbol for gold?",
            options=["Ag", "Au", "Gd", "Go"],
            correct_answer="B",
            subject="science",
            topic="chemistry"
        )
        
        result = classifier.predict(test_question)
        print(f"\nClassification Result:")
        print(f"Subject: {result.predicted_subject} (confidence: {result.subject_confidence:.3f})")
        print(f"Topic: {result.predicted_topic} (confidence: {result.topic_confidence:.3f})")
        print(f"Difficulty: {result.predicted_difficulty} (confidence: {result.difficulty_confidence:.3f})")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("This is expected with small sample data. In production, use larger datasets.")