"""
Data Ingestion Pipeline for Government Exam Data
Automated data collection, preprocessing, and preparation for ML training
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExamData:
    """Structure for exam question data"""
    exam_code: str
    question_id: str
    question_text: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str]
    subject: str
    topic: str
    difficulty: str
    year: int
    stage: str
    source: str
    tags: List[str]

@dataclass
class ExamPaper:
    """Structure for complete exam papers"""
    exam_code: str
    year: int
    stage: str
    subject: str
    total_questions: int
    total_marks: int
    duration: str
    questions: List[ExamData]
    metadata: Dict

class DataSourceManager:
    """Manages data sources and collection strategies"""
    
    def __init__(self, config_path: str = "config/data_sources.json"):
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _load_config(self, config_path: str) -> Dict:
        """Load data source configuration"""
        default_config = {
            "free_exam_websites": [
                "https://gradeup.co",
                "https://unacademy.com", 
                "https://byjus.com",
                "https://testbook.com",
                "https://affairscloud.com"
            ],
            "paid_exam_websites": [
                "https://visionias.in",
                "https://nextiasacademy.in",
                "https://coachingselective.com"
            ],
            "official_sources": [
                "https://ssc.nic.in",
                "https://upsconline.nic.in",
                "https://ibps.in",
                "https://rrbcdg.gov.in"
            ],
            "repository_urls": {
                "upsc_papers": "https://upsc.gov.in/exam/syllabus/",
                "ssc_papers": "https://ssc.nic.in/WebPortal/ViewQuestionPaper.aspx",
                "banking_papers": "https://ibps.in/crp_po_rrb/POX/",
                "railway_papers": "https://rrbcdg.gov.in/view_exam.aspx"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config

class ExamDataExtractor:
    """Extracts exam data from various sources"""
    
    def __init__(self, data_manager: DataSourceManager):
        self.data_manager = data_manager
        self.output_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    async def extract_from_web_source(self, url: str, exam_code: str) -> List[ExamData]:
        """Extract exam data from a web source"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._parse_html_content(html_content, exam_code)
                    else:
                        logger.warning(f"Failed to fetch {url}: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error extracting from {url}: {str(e)}")
            return []
    
    def _parse_html_content(self, html_content: str, exam_code: str) -> List[ExamData]:
        """Parse HTML content to extract exam questions"""
        # This is a simplified version - in production, you'd use BeautifulSoup
        # and more sophisticated parsing logic based on website structure
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        questions = []
        # Parse logic would depend on specific website structure
        # This is a template implementation
        
        return questions
    
    def extract_from_pdf(self, pdf_path: str, exam_code: str) -> List[ExamData]:
        """Extract exam data from PDF files"""
        try:
            import PyPDF2
            from pdfplumber import PDF
            
            questions = []
            
            # Method 1: Using pdfplumber for better text extraction
            with PDF(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    page_questions = self._parse_pdf_text(text, exam_code, page_num + 1)
                    questions.extend(page_questions)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {str(e)}")
            return []
    
    def _parse_pdf_text(self, text: str, exam_code: str, page_num: int) -> List[ExamData]:
        """Parse extracted PDF text into structured exam data"""
        # This is a template - implement specific parsing logic
        # based on exam paper format
        
        questions = []
        
        # Split text into potential questions
        # Look for question markers, options, etc.
        
        return questions
    
    def extract_from_database(self, db_path: str, query: str) -> List[ExamData]:
        """Extract exam data from local database"""
        try:
            import sqlite3
            
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
                
            questions = []
            for _, row in df.iterrows():
                question = ExamData(
                    exam_code=row.get('exam_code', exam_code),
                    question_id=row.get('question_id', ''),
                    question_text=row.get('question_text', ''),
                    options=json.loads(row.get('options', '[]')),
                    correct_answer=row.get('correct_answer', ''),
                    explanation=row.get('explanation'),
                    subject=row.get('subject', ''),
                    topic=row.get('topic', ''),
                    difficulty=row.get('difficulty', 'medium'),
                    year=row.get('year', 2024),
                    stage=row.get('stage', 'tier1'),
                    source=row.get('source', 'database'),
                    tags=json.loads(row.get('tags', '[]'))
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error extracting from database: {str(e)}")
            return []

class DataPreprocessor:
    """Preprocesses and cleans exam data for ML training"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.tokenizer = None  # Will be initialized with spaCy or NLTK
    
    def _load_stop_words(self) -> set:
        """Load common stop words for text preprocessing"""
        # English and Hindi stop words for government exam context
        return {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'through',
            
            # Hindi
            'कि', 'के', 'का', 'की', 'से', 'में', 'पर', 'और', 'या', 'इस', 'उस',
            'है', 'हैं', 'हो', 'होगा', 'होगी', 'कर', 'करता', 'करती', 'करें',
            
            # Exam specific
            'option', 'choices', 'select', 'choose', 'which', 'what', 'where',
            'when', 'why', 'how', 'answer', 'question', 'correct', 'wrong'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        return text.strip()
    
    def normalize_options(self, options: List[str]) -> List[str]:
        """Normalize multiple choice options"""
        if not options:
            return []
        
        normalized = []
        for option in options:
            # Remove common prefixes like (A), 1), A., etc.
            clean_option = re.sub(r'^\s*[\(\[]?\s*[a-zA-Z0-9]+\s*[\)\]]?\s*[\.\:]?\s*', '', option)
            normalized.append(self.clean_text(clean_option))
        
        return normalized
    
    def extract_features(self, question: ExamData) -> Dict:
        """Extract features from question for ML training"""
        features = {
            # Text features
            'question_length': len(question.question_text),
            'question_word_count': len(question.question_text.split()),
            'options_count': len(question.options),
            
            # Subject encoding
            'subject_encoded': self._encode_subject(question.subject),
            
            # Difficulty encoding
            'difficulty_encoded': self._encode_difficulty(question.difficulty),
            
            # Topic embedding (placeholder - would use word embeddings)
            'topic_vector': self._get_topic_vector(question.topic),
            
            # Temporal features
            'year_normalized': (question.year - 2010) / 15,  # Normalize 2010-2024 range
            
            # Source reliability
            'source_reliability': self._get_source_reliability(question.source)
        }
        
        return features
    
    def _encode_subject(self, subject: str) -> int:
        """Encode subject as integer for ML models"""
        subject_mapping = {
            'english': 1, 'mathematics': 2, 'reasoning': 3, 'general_knowledge': 4,
            'quantitative_aptitude': 5, 'general_awareness': 6, 'computer_knowledge': 7,
            'history': 8, 'geography': 9, 'polity': 10, 'economy': 11,
            'science': 12, 'current_affairs': 13
        }
        return subject_mapping.get(subject.lower(), 0)
    
    def _encode_difficulty(self, difficulty: str) -> int:
        """Encode difficulty level"""
        mapping = {'easy': 1, 'low': 1, 'medium': 2, 'moderate': 2, 'high': 3, 'hard': 3, 'very_high': 4}
        return mapping.get(difficulty.lower(), 2)
    
    def _get_topic_vector(self, topic: str) -> List[float]:
        """Get topic vector (placeholder for word embeddings)"""
        # In production, use actual word embeddings like Word2Vec, GloVe, or BERT
        # This is a simplified placeholder
        return [0.0] * 50  # 50-dimensional vector
    
    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for data source"""
        reliability_mapping = {
            'official': 1.0, 'paid': 0.9, 'free': 0.7, 'user_generated': 0.5
        }
        source_type = source.split('/')[2] if 'http' in source else 'unknown'
        
        if 'nic.in' in source_type or 'gov.in' in source_type:
            return 1.0
        elif 'vision' in source_type or 'nextias' in source_type:
            return 0.9
        elif 'gradeup' in source_type or 'unacademy' in source_type:
            return 0.8
        else:
            return 0.6
    
    def prepare_training_data(self, questions: List[ExamData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        features = []
        labels = []
        
        for question in questions:
            feature_dict = self.extract_features(question)
            features.append(list(feature_dict.values()))
            
            # Label encoding for correct answer (A, B, C, D -> 0, 1, 2, 3)
            label = ord(question.correct_answer.upper()) - ord('A') if question.correct_answer else 0
            labels.append(label)
        
        return np.array(features), np.array(labels)

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/data_sources.json"):
        self.data_manager = DataSourceManager(config_path)
        self.extractor = ExamDataExtractor(self.data_manager)
        self.preprocessor = DataPreprocessor()
        
    def run_extraction_pipeline(self, sources: List[str], exam_codes: List[str]) -> str:
        """Run complete data extraction pipeline"""
        logger.info("Starting data extraction pipeline...")
        
        all_questions = []
        
        # Extract from each source
        for source in sources:
            for exam_code in exam_codes:
                logger.info(f"Extracting {exam_code} from {source}")
                
                if source.endswith('.pdf'):
                    questions = self.extractor.extract_from_pdf(source, exam_code)
                elif 'http' in source:
                    # Web scraping - would need async implementation
                    questions = []
                else:
                    questions = self.extractor.extract_from_database(source, f"SELECT * FROM {exam_code}")
                
                all_questions.extend(questions)
        
        # Save raw data
        raw_data_path = self._save_raw_data(all_questions)
        
        # Preprocess data
        processed_data_path = self._preprocess_data(all_questions)
        
        logger.info(f"Pipeline completed. Raw data: {raw_data_path}, Processed data: {processed_data_path}")
        return processed_data_path
    
    def _save_raw_data(self, questions: List[ExamData]) -> str:
        """Save raw extracted data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_exam_data_{timestamp}.json"
        filepath = self.extractor.output_dir / filename
        
        data = []
        for q in questions:
            data.append({
                'exam_code': q.exam_code,
                'question_id': q.question_id,
                'question_text': q.question_text,
                'options': q.options,
                'correct_answer': q.correct_answer,
                'explanation': q.explanation,
                'subject': q.subject,
                'topic': q.topic,
                'difficulty': q.difficulty,
                'year': q.year,
                'stage': q.stage,
                'source': q.source,
                'tags': q.tags
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def _preprocess_data(self, questions: List[ExamData]) -> str:
        """Preprocess and save cleaned data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_exam_data_{timestamp}.json"
        filepath = self.extractor.processed_dir / filename
        
        processed_data = []
        
        for q in questions:
            # Clean and normalize data
            cleaned_question = ExamData(
                exam_code=q.exam_code,
                question_id=q.question_id,
                question_text=self.preprocessor.clean_text(q.question_text),
                options=self.preprocessor.normalize_options(q.options),
                correct_answer=q.correct_answer.upper() if q.correct_answer else '',
                explanation=self.preprocessor.clean_text(q.explanation) if q.explanation else '',
                subject=q.subject.lower(),
                topic=q.topic.lower(),
                difficulty=q.difficulty.lower(),
                year=q.year,
                stage=q.stage.lower(),
                source=q.source,
                tags=[tag.lower() for tag in q.tags]
            )
            
            # Extract features
            features = self.preprocessor.extract_features(cleaned_question)
            
            processed_data.append({
                'question': cleaned_question.__dict__,
                'features': features
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Define extraction sources and target exams
    sources = [
        "data/sample_exams.db",  # Local database
        "data/sample_papers.pdf"  # PDF papers
    ]
    
    exam_codes = ["ssc_cgl", "ibps_po", "upsc_cse"]
    
    # Run extraction
    processed_file = pipeline.run_extraction_pipeline(sources, exam_codes)
    print(f"Data processing completed. Output: {processed_file}")