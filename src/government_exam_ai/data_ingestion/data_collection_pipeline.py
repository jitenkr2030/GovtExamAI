"""
Data Collection and Preprocessing Pipeline for Government Exam Training Data
Author: MiniMax Agent
Date: 2025-12-01

This module provides comprehensive data collection capabilities for building
training datasets from PDF exam papers, text files, and structured data sources.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
from datetime import datetime
import pickle

# PDF processing libraries
try:
    import PyPDF2
    from pdfplumber import PDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF processing libraries not available. Install PyPDF2 and pdfplumber for PDF support.")

# Text processing
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Spacy for advanced NLP
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install spaCy for advanced NLP capabilities.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Advanced text preprocessing for exam data."""
    
    def __init__(self, use_spacy=True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.lemmatizer = WordNetLemmatizer()
        
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
            
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy English model not found. Using basic preprocessing.")
                self.use_spacy = False
                
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
            
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        if self.use_spacy:
            doc = self.nlp(text)
            return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Basic tokenization and lemmatization
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in stop_words and token.isalpha()]
            return tokens
        
    def extract_key_features(self, text: str) -> Dict[str, Any]:
        """Extract key features from text for classification."""
        text = self.clean_text(text)
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'has_math_symbols': bool(re.search(r'[+\-*/=<>≤≥∑∫√]', text)),
            'has_years': bool(re.search(r'\b(19|20)\d{2}\b', text)),
            'has_numbers': bool(re.search(r'\d+', text)),
            'subject_markers': self._extract_subject_markers(text),
            'difficulty_indicators': self._extract_difficulty_indicators(text)
        }
        
        return features
        
    def _extract_subject_markers(self, text: str) -> List[str]:
        """Extract subject-specific markers from text."""
        markers = []
        text_lower = text.lower()
        
        subject_patterns = {
            'mathematics': [r'\b(calculate|compute|solve|equation|algebra|geometry|calculus)\b'],
            'science': [r'\b(experiment|theory|hypothesis|observe|measure|laboratory)\b'],
            'history': [r'\b(battle|treaty|emperor|kingdom|ancient|medieval|century)\b'],
            'geography': [r'\b(mountain|river|climate|continent|country|ocean|map)\b'],
            'polity': [r'\b(constitution|parliament|democracy|law|rights|court|legislature)\b'],
            'economics': [r'\b(market|economy|inflation|interest|trade|fiscal|monetary)\b']
        }
        
        for subject, patterns in subject_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    markers.append(subject)
                    break
                    
        return markers
        
    def _extract_difficulty_indicators(self, text: str) -> Dict[str, float]:
        """Extract difficulty indicators from text."""
        text_lower = text.lower()
        
        indicators = {
            'complexity_score': 0.0,
            'technical_terms': 0,
            'abstract_concepts': 0
        }
        
        # Calculate complexity based on various factors
        words = text.split()
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sent_tokenize(text)])
        
        # Technical terms (words longer than 8 characters)
        technical_terms = [word for word in words if len(word) > 8]
        indicators['technical_terms'] = len(technical_terms) / len(words) if words else 0
        
        # Abstract concepts
        abstract_words = ['concept', 'theory', 'principle', 'method', 'process', 'system']
        abstract_count = sum(1 for word in words if word.lower() in abstract_words)
        indicators['abstract_concepts'] = abstract_count / len(words) if words else 0
        
        # Overall complexity score
        indicators['complexity_score'] = (
            (avg_sentence_length / 20.0) * 0.4 +  # Sentence length factor
            indicators['technical_terms'] * 0.3 +  # Technical terms factor
            indicators['abstract_concepts'] * 0.3   # Abstract concepts factor
        )
        
        return indicators


class PDFDataExtractor:
    """Extract structured data from PDF exam papers."""
    
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.pdf_text_cache = {}
        
    def extract_from_pdf(self, pdf_path: str, extraction_rules: Optional[Dict] = None) -> List[Dict]:
        """Extract exam questions from PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing not available. Install PyPDF2 and pdfplumber.")
            
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Extracting data from PDF: {pdf_path}")
        
        # Extract text using pdfplumber (better for structured documents)
        with PDF(pdf_path) as pdf:
            text_content = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
        full_text = '\n'.join(text_content)
        
        # Apply extraction rules
        if extraction_rules is None:
            extraction_rules = self._get_default_extraction_rules()
            
        questions = self._extract_questions_from_text(full_text, extraction_rules)
        
        # Process each question
        processed_questions = []
        for q in questions:
            processed_q = self._process_extracted_question(q)
            if processed_q:
                processed_questions.append(processed_q)
                
        logger.info(f"Extracted {len(processed_questions)} questions from {pdf_path}")
        return processed_questions
        
    def _get_default_extraction_rules(self) -> Dict:
        """Get default extraction rules for common exam formats."""
        return {
            'question_patterns': [
                r'Q\.?\d*\.?\s*(.*?)(?=Q\.?\d*\.?|\n\n|$)',
                r'^\d+\.\s*(.*?)(?=^\d+\.|\n\n|$)',
                r'\*\*(.*?)\*\*',  # Bold text questions
                r'\d+\)\s*(.*?)(?=\d+\)|$)'
            ],
            'option_patterns': [
                r'[A-D]\)\s*(.*?)(?=[A-D]\)|$)',
                r'[A-D]\.\s*(.*?)(?=[A-D]\.|$)'
            ],
            'answer_markers': [
                r'Answer[:\s]*(.*?)(?=\n|$)',
                r'Solution[:\s]*(.*?)(?=\n|$)'
            ]
        }
        
    def _extract_questions_from_text(self, text: str, rules: Dict) -> List[Dict]:
        """Extract questions using regex patterns."""
        questions = []
        
        # Split text into sections
        sections = re.split(r'\n\s*\n', text)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Look for question patterns
            for pattern in rules['question_patterns']:
                matches = re.findall(pattern, section, re.MULTILINE | re.DOTALL)
                for match in matches:
                    question_text = self.preprocessor.clean_text(match.strip())
                    if len(question_text) > 10:  # Filter out very short matches
                        
                        question_data = {
                            'question_text': question_text,
                            'source': 'pdf_extraction',
                            'extracted_timestamp': datetime.now().isoformat()
                        }
                        
                        # Try to extract options
                        options = self._extract_options(section, rules['option_patterns'])
                        if options:
                            question_data['options'] = options
                            
                        # Try to extract answer
                        answer = self._extract_answer(section, rules['answer_markers'])
                        if answer:
                            question_data['answer_text'] = answer
                            
                        questions.append(question_data)
                        
        return questions
        
    def _extract_options(self, text: str, option_patterns: List[str]) -> Optional[List[str]]:
        """Extract multiple choice options from text."""
        options = []
        for pattern in option_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                option_text = self.preprocessor.clean_text(match.strip())
                if option_text and len(option_text) > 1:
                    options.append(option_text)
            if options:
                break
        return options if len(options) >= 2 else None
        
    def _extract_answer(self, text: str, answer_patterns: List[str]) -> Optional[str]:
        """Extract answer from text."""
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = self.preprocessor.clean_text(match.group(1).strip())
                if answer_text:
                    return answer_text
        return None
        
    def _process_extracted_question(self, question_data: Dict) -> Optional[Dict]:
        """Process and enhance extracted question data."""
        try:
            question_text = question_data['question_text']
            
            # Extract features
            features = self.preprocessor.extract_key_features(question_text)
            
            # Determine subject based on markers and content
            subject = self._classify_subject(question_text, features)
            topic = self._classify_topic(question_text, features, subject)
            difficulty = self._classify_difficulty(features)
            
            processed_question = {
                **question_data,
                'subject': subject,
                'topic': topic,
                'difficulty': difficulty,
                'features': features,
                'text_hash': hashlib.md5(question_text.encode()).hexdigest()
            }
            
            return processed_question
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return None
            
    def _classify_subject(self, text: str, features: Dict) -> str:
        """Classify subject based on text content and markers."""
        text_lower = text.lower()
        
        # Use subject markers from features
        markers = features.get('subject_markers', [])
        if markers:
            return markers[0]  # Return first marker
            
        # Pattern-based classification
        subject_patterns = {
            'Mathematics': [
                r'\b(calculate|compute|solve|equation|algebra|geometry|calculus|math)\b',
                r'[+\-*/=<>≤≥∑∫√]', r'\b\d+[\s]*[+\-*/=][\s]*\d+\b'
            ],
            'Physics': [
                r'\b(force|energy|motion|velocity|acceleration|pressure|voltage|current)\b',
                r'\b(newton|watt|joule|ohm|ampere|volt)\b'
            ],
            'Chemistry': [
                r'\b(element|compound|molecule|reaction|atom|periodic|chemical)\b',
                r'\b(hydrogen|oxygen|nitrogen|carbon|sodium|chlorine)\b'
            ],
            'Biology': [
                r'\b(cell|gene|organism|species|photosynthesis|respiration|circulation)\b',
                r'\b(plant|animal|bacteria|virus|protein|enzymes)\b'
            ],
            'History': [
                r'\b(battle|treaty|emperor|kingdom|ancient|medieval|century|revolution)\b',
                r'\b(british|independence|partition|war|freedom)\b'
            ],
            'Geography': [
                r'\b(mountain|river|climate|continent|country|ocean|map|geography)\b',
                r'\b(india|world|earth|land|climate|weather)\b'
            ],
            'Polity': [
                r'\b(constitution|parliament|democracy|law|rights|court|legislature)\b',
                r'\b(election|voting|congress|parliament|bill|act)\b'
            ],
            'Economics': [
                r'\b(market|economy|inflation|interest|trade|fiscal|monetary|bank)\b',
                r'\b(gdp|revenue|budget|income|export|import)\b'
            ]
        }
        
        scores = {}
        for subject, patterns in subject_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[subject] = score
            
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'General Knowledge'  # Default subject
            
    def _classify_topic(self, text: str, features: Dict, subject: str) -> str:
        """Classify topic within the subject."""
        # This is a simplified topic classification
        # In a real implementation, you'd have more sophisticated topic models
        
        text_lower = text.lower()
        
        topic_mapping = {
            'Mathematics': {
                'arithmetic': [r'\b(add|subtract|multiply|divide|percentage|fraction)\b'],
                'algebra': [r'\b(equation|variable|linear|quadratic|polynomial)\b'],
                'geometry': [r'\b(triangle|circle|square|angle|area|perimeter)\b'],
                'calculus': [r'\b(derivative|integral|limit|function|slope)\b']
            },
            'Physics': {
                'mechanics': [r'\b(force|motion|velocity|acceleration|mass|inertia)\b'],
                'thermodynamics': [r'\b(heat|temperature|energy|entropy|thermodynamic)\b'],
                'electromagnetism': [r'\b(electric|magnetic|voltage|current|resistance)\b'],
                'optics': [r'\b(light|ray|reflection|refraction|lens|mirror)\b']
            },
            'Chemistry': {
                'organic': [r'\b(organic|carbon|hydrocarbon|alcohol|aldehyde|ketone)\b'],
                'inorganic': [r'\b(inorganic|salt|acid|base|metal|non-metal)\b'],
                'physical': [r'\b(physical|state|matter|phase|phase change)\b'],
                'analytical': [r'\b(analysis|test|detection|identification|quantitative)\b']
            }
        }
        
        if subject in topic_mapping:
            scores = {}
            for topic, patterns in topic_mapping[subject].items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        score += 1
                scores[topic] = score
                
            if scores:
                return max(scores.items(), key=lambda x: x[1])[0]
                
        return f'{subject} General'
        
    def _classify_difficulty(self, features: Dict) -> str:
        """Classify difficulty level based on features."""
        complexity_score = features.get('complexity_score', 0)
        technical_terms = features.get('technical_terms', 0)
        avg_word_length = features.get('avg_word_length', 0)
        
        # Difficulty scoring algorithm
        difficulty_score = (
            complexity_score * 0.4 +
            technical_terms * 0.3 +
            (avg_word_length / 10.0) * 0.3
        )
        
        if difficulty_score < 0.3:
            return 'Easy'
        elif difficulty_score < 0.6:
            return 'Medium'
        else:
            return 'Hard'


class DataAugmentor:
    """Data augmentation for increasing training dataset size."""
    
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        
    def augment_questions(self, questions: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """Augment question dataset using various techniques."""
        logger.info(f"Augmenting {len(questions)} questions with factor {augmentation_factor}")
        
        augmented_questions = []
        
        for question in questions:
            # Add original question
            augmented_questions.append(question)
            
            # Generate augmented versions
            for _ in range(augmentation_factor - 1):
                augmented_q = self._apply_augmentation(question)
                if augmented_q:
                    augmented_questions.append(augmented_q)
                    
        logger.info(f"Created {len(augmented_questions)} augmented questions")
        return augmented_questions
        
    def _apply_augmentation(self, question: Dict) -> Optional[Dict]:
        """Apply a single augmentation technique."""
        question_text = question['question_text']
        
        # Choose augmentation technique
        techniques = [
            self._paraphrase_question,
            self._add_synonyms,
            self._modify_numbers,
            self._change_sentence_structure
        ]
        
        technique = np.random.choice(techniques)
        augmented_text = technique(question_text)
        
        if augmented_text and augmented_text != question_text:
            augmented_question = question.copy()
            augmented_question['question_text'] = augmented_text
            augmented_question['source'] = 'augmented'
            augmented_question['augmented_timestamp'] = datetime.now().isoformat()
            return augmented_question
            
        return None
        
    def _paraphrase_question(self, text: str) -> str:
        """Simple paraphrasing using synonym replacement."""
        # This is a simplified paraphrasing approach
        # In practice, you'd use more sophisticated NLP models
        
        replacements = {
            'what': 'which',
            'how': 'in what way',
            'why': 'for what reason',
            'when': 'at what time',
            'where': 'at what place',
            'explain': 'describe',
            'define': 'explain',
            'calculate': 'compute',
            'find': 'determine',
            'show': 'demonstrate'
        }
        
        text_lower = text.lower()
        for word, replacement in replacements.items():
            if re.search(r'\b' + word + r'\b', text_lower):
                return re.sub(r'\b' + word + r'\b', replacement, text, flags=re.IGNORECASE)
                
        return text
        
    def _add_synonyms(self, text: str) -> str:
        """Add synonyms to increase vocabulary diversity."""
        # Simplified synonym addition
        synonyms = {
            'important': 'significant',
            'large': 'substantial',
            'small': 'minimal',
            'good': 'beneficial',
            'bad': 'detrimental',
            'quick': 'rapid',
            'slow': 'gradual'
        }
        
        text_parts = text.split()
        if len(text_parts) > 3 and np.random.random() < 0.3:
            # Replace a random word with its synonym
            replace_idx = np.random.randint(1, len(text_parts) - 1)
            word = text_parts[replace_idx].lower()
            
            for orig, synonym in synonyms.items():
                if word == orig.lower():
                    text_parts[replace_idx] = synonym.capitalize() if text_parts[replace_idx][0].isupper() else synonym
                    break
                    
        return ' '.join(text_parts)
        
    def _modify_numbers(self, text: str) -> str:
        """Modify numbers in the question."""
        # Find and modify numbers
        number_matches = re.finditer(r'\b\d+\b', text)
        
        for match in number_matches:
            original_num = int(match.group())
            # Add or subtract a small random number
            modified_num = original_num + np.random.randint(-5, 6)
            if modified_num > 0:
                text = text[:match.start()] + str(modified_num) + text[match.end():]
                
        return text
        
    def _change_sentence_structure(self, text: str) -> str:
        """Change basic sentence structure."""
        # Simple structure changes
        if 'is' in text.lower():
            text = re.sub(r'\b(is|are|was|were)\b', 'becomes', text, flags=re.IGNORECASE)
        elif 'are' in text.lower():
            text = re.sub(r'\bare\b', 'exist as', text, flags=re.IGNORECASE)
            
        return text


class DatasetBuilder:
    """Main class for building comprehensive training datasets."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.preprocessor = DataPreprocessor()
        self.pdf_extractor = PDFDataExtractor(self.preprocessor)
        self.augmentor = DataAugmentor(self.preprocessor)
        
        # Data storage
        self.raw_data = []
        self.processed_data = []
        self.dataset_stats = {}
        
    def add_data_source(self, source_path: str, source_type: str, 
                       extraction_config: Optional[Dict] = None) -> None:
        """Add a new data source (PDF, text file, or JSON)."""
        logger.info(f"Adding data source: {source_path} (type: {source_type})")
        
        source_path = Path(source_path)
        
        if not source_path.exists():
            logger.error(f"Data source not found: {source_path}")
            return
            
        try:
            if source_type.lower() == 'pdf':
                extracted_data = self.pdf_extractor.extract_from_pdf(
                    str(source_path), extraction_config
                )
                self.raw_data.extend(extracted_data)
                
            elif source_type.lower() == 'json':
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.raw_data.extend(data)
                
            elif source_type.lower() == 'csv':
                df = pd.read_csv(source_path)
                self.raw_data.extend(df.to_dict('records'))
                
            elif source_type.lower() == 'text':
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Simple text processing - split into paragraphs
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    if len(para) > 20:  # Filter very short texts
                        self.raw_data.append({
                            'question_text': para,
                            'source': 'text_file',
                            'extracted_timestamp': datetime.now().isoformat()
                        })
                        
            else:
                logger.error(f"Unsupported source type: {source_type}")
                # Log the number of items added
                try:
                    if source_type.lower() == 'pdf':
                        logger.info(f"Added {len(extracted_data)} items from {source_path}")
                    elif source_type.lower() == 'text':
                        logger.info(f"Added {len(paragraphs)} items from {source_path}")
                    elif source_type.lower() == 'json':
                        logger.info(f"Added {len(data)} items from {source_path}")
                    elif source_type.lower() == 'csv':
                        logger.info(f"Added {len(df)} items from {source_path}")
                except NameError:
                    logger.info(f"Added items from {source_path}")
            
        except Exception as e:
            logger.error(f"Error processing data source {source_path}: {e}")
            
    def process_and_enhance_data(self, min_text_length: int = 20, 
                                deduplicate: bool = True) -> pd.DataFrame:
        """Process and enhance all collected data."""
        logger.info("Processing and enhancing collected data")
        
        processed_questions = []
        seen_hashes = set()
        
        for item in self.raw_data:
            try:
                if isinstance(item, dict) and 'question_text' in item:
                    question_text = item['question_text']
                    
                    # Filter by length
                    if len(question_text.strip()) < min_text_length:
                        continue
                        
                    # Deduplication
                    if deduplicate:
                        text_hash = hashlib.md5(question_text.encode()).hexdigest()
                        if text_hash in seen_hashes:
                            continue
                        seen_hashes.add(text_hash)
                        
                    # Extract features
                    features = self.preprocessor.extract_key_features(question_text)
                    
                    # Enhanced question data
                    enhanced_question = {
                        'question_text': question_text.strip(),
                        'source': item.get('source', 'unknown'),
                        'extracted_timestamp': item.get('extracted_timestamp', datetime.now().isoformat()),
                        'features': features,
                        'text_length': len(question_text),
                        'word_count': len(question_text.split())
                    }
                    
                    # Copy any additional fields
                    for key, value in item.items():
                        if key not in enhanced_question:
                            enhanced_question[key] = value
                            
                    processed_questions.append(enhanced_question)
                    
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
                
        self.processed_data = processed_questions
        self._calculate_dataset_stats()
        
        logger.info(f"Processed {len(processed_questions)} questions")
        return pd.DataFrame(processed_questions)
        
    def augment_dataset(self, augmentation_factor: int = 2) -> pd.DataFrame:
        """Augment the processed dataset."""
        if not self.processed_data:
            logger.warning("No processed data to augment")
            return pd.DataFrame()
            
        logger.info(f"Augmenting dataset with factor {augmentation_factor}")
        augmented_data = self.augmentor.augment_questions(self.processed_data, augmentation_factor)
        self.processed_data = augmented_data
        self._calculate_dataset_stats()
        
        return pd.DataFrame(augmented_data)
        
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate comprehensive dataset statistics."""
        if not self.processed_data:
            self.dataset_stats = {}
            return {}
            
        df = pd.DataFrame(self.processed_data)
        
        stats = {
            'total_questions': len(df),
            'sources': df['source'].value_counts().to_dict(),
            'subjects': df.get('subject', pd.Series()).value_counts().to_dict(),
            'topics': df.get('topic', pd.Series()).value_counts().to_dict(),
            'difficulty_levels': df.get('difficulty', pd.Series()).value_counts().to_dict(),
            'text_statistics': {
                'avg_length': df['text_length'].mean(),
                'min_length': df['text_length'].min(),
                'max_length': df['text_length'].max(),
                'avg_words': df['word_count'].mean()
            },
            'created_timestamp': datetime.now().isoformat()
        }
        
        self.dataset_stats = stats
        return stats
        
    def save_dataset(self, save_path: str, include_stats: bool = True) -> None:
        """Save the complete dataset."""
        if not self.processed_data:
            logger.warning("No data to save")
            return
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(save_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            
        # Save as CSV
        df = pd.DataFrame(self.processed_data)
        df.to_csv(save_path.with_suffix('.csv'), index=False, encoding='utf-8')
        
        # Save statistics (convert numpy types to regular Python types)
        if include_stats:
            # Convert numpy types to regular Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            clean_stats = convert_numpy_types(self.dataset_stats)
            with open(save_path.with_suffix('.stats.json'), 'w') as f:
                json.dump(clean_stats, f, indent=2)
                
        logger.info(f"Dataset saved to {save_path}")
        logger.info(f"Dataset statistics: {self.dataset_stats}")
        
    def load_dataset(self, load_path: str) -> None:
        """Load a previously saved dataset."""
        load_path = Path(load_path)
        
        if load_path.suffix == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
        else:
            df = pd.read_csv(load_path)
            self.processed_data = df.to_dict('records')
            
        self._calculate_dataset_stats()
        logger.info(f"Loaded dataset from {load_path}")
        
    def get_training_ready_data(self, test_size: float = 0.2, 
                               validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        if not self.processed_data:
            raise ValueError("No processed data available")
            
        df = pd.DataFrame(self.processed_data)
        
        # First split: train + temp (validation + test)
        train_df, temp_df = train_test_split(df, test_size=(validation_size + test_size), 
                                           random_state=42, stratify=df.get('subject'))
        
        # Second split: validation and test
        val_df, test_df = train_test_split(temp_df, test_size=test_size/(validation_size + test_size),
                                          random_state=42, stratify=temp_df.get('subject'))
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df


# Example usage and testing
if __name__ == "__main__":
    # Initialize dataset builder
    builder = DatasetBuilder()
    
    # Example: Add data sources (would need actual files to test)
    print("Dataset builder initialized")
    print("Available methods:")
    print("- add_data_source(): Add PDF, JSON, CSV, or text files")
    print("- process_and_enhance_data(): Clean and preprocess data")
    print("- augment_dataset(): Increase dataset size with augmentation")
    print("- save_dataset(): Save processed dataset")
    print("- get_training_ready_data(): Split into train/val/test sets")
    
    # Sample data creation for testing
    sample_questions = [
        {
            'question_text': 'What is the capital of India?',
            'subject': 'Geography',
            'topic': 'Indian Geography',
            'difficulty': 'Easy'
        },
        {
            'question_text': 'Calculate the derivative of x^2 + 3x + 2',
            'subject': 'Mathematics',
            'topic': 'Calculus',
            'difficulty': 'Medium'
        }
    ]
    
    builder.raw_data = sample_questions
    processed_df = builder.process_and_enhance_data()
    print(f"Processed {len(processed_df)} sample questions")
    print(processed_df.head())