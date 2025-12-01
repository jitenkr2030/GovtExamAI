"""
Analytics and Performance Tracking System
Comprehensive analytics for government exam preparation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a student"""
    student_id: str
    test_id: str
    total_marks: float
    max_marks: float
    percentage: float
    rank: Optional[int]
    percentile: float
    subject_scores: Dict[str, float]
    difficulty_scores: Dict[str, float]
    time_taken: float
    accuracy_by_topic: Dict[str, float]
    weak_areas: List[str]
    strong_areas: List[str]
    improvement_suggestions: List[str]

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    student_id: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    overall_performance: Dict
    subject_analysis: Dict
    difficulty_analysis: Dict
    temporal_analysis: Dict
    comparative_analysis: Dict
    predictions: Dict
    recommendations: Dict

class DataAggregator:
    """Aggregates and processes performance data"""
    
    def __init__(self, data_path: str = "data/performance"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.performance_data = []
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load performance data from all available sources"""
        # Load from generated tests
        tests_dir = Path("tests/generated")
        if tests_dir.exists():
            for test_file in tests_dir.glob("*.json"):
                self._load_test_results(test_file)
        
        # Load from exam results
        results_dir = Path("data/exam_results")
        if results_dir.exists():
            for result_file in results_dir.glob("*.json"):
                self._load_exam_results(result_file)
        
        logger.info(f"Loaded {len(self.performance_data)} performance records")
    
    def _load_test_results(self, test_file: Path):
        """Load results from generated tests"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # This would need to be adapted based on actual test result format
            # For now, creating a placeholder structure
            performance_record = {
                'student_id': test_data.get('student_id', 'unknown'),
                'test_id': test_data.get('test_id', ''),
                'exam_code': test_data.get('specification', {}).get('exam_code', ''),
                'total_marks': 0,  # Would be calculated from results
                'max_marks': test_data.get('specification', {}).get('total_marks', 0),
                'percentage': 0,  # Would be calculated from actual scores
                'test_date': test_data.get('generated_at', datetime.now().isoformat()),
                'subject_scores': {},
                'time_taken': 0  # Would be actual time taken
            }
            
            self.performance_data.append(performance_record)
            
        except Exception as e:
            logger.error(f"Error loading test results from {test_file}: {str(e)}")
    
    def _load_exam_results(self, result_file: Path):
        """Load actual exam results"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Process based on actual exam result format
            self.performance_data.append(result_data)
            
        except Exception as e:
            logger.error(f"Error loading exam results from {result_file}: {str(e)}")
    
    def get_student_data(self, student_id: str) -> List[Dict]:
        """Get all data for a specific student"""
        return [record for record in self.performance_data 
                if record.get('student_id') == student_id]
    
    def get_cohort_data(self, filters: Dict = None) -> List[Dict]:
        """Get data for a cohort of students"""
        cohort_data = self.performance_data
        
        if filters:
            if 'exam_code' in filters:
                cohort_data = [r for r in cohort_data if r.get('exam_code') == filters['exam_code']]
            if 'date_range' in filters:
                start_date, end_date = filters['date_range']
                cohort_data = [r for r in cohort_data 
                             if start_date <= datetime.fromisoformat(r.get('test_date', '')) <= end_date]
        
        return cohort_data

class PerformanceAnalyzer:
    """Analyzes student and cohort performance"""
    
    def __init__(self, data_aggregator: DataAggregator):
        self.data_aggregator = data_aggregator
        self.scaler = StandardScaler()
    
    def analyze_individual_performance(self, student_id: str, 
                                     days_back: int = 30) -> PerformanceMetrics:
        """Analyze individual student performance"""
        student_data = self.data_aggregator.get_student_data(student_id)
        
        if not student_data:
            raise ValueError(f"No data found for student {student_id}")
        
        # Filter data by time period
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = [
            record for record in student_data 
            if datetime.fromisoformat(record.get('test_date', '')) >= cutoff_date
        ]
        
        if not recent_data:
            recent_data = student_data  # Use all data if no recent data
        
        # Calculate overall metrics
        total_marks = sum(record.get('total_marks', 0) for record in recent_data)
        max_marks = sum(record.get('max_marks', 0) for record in recent_data)
        overall_percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0
        
        # Subject-wise analysis
        subject_scores = defaultdict(list)
        for record in recent_data:
            for subject, score in record.get('subject_scores', {}).items():
                subject_scores[subject].append(score)
        
        avg_subject_scores = {
            subject: np.mean(scores) 
            for subject, scores in subject_scores.items()
        }
        
        # Difficulty analysis (would need difficulty information in records)
        difficulty_scores = {'easy': 0, 'medium': 0, 'hard': 0}  # Placeholder
        
        # Identify weak and strong areas
        sorted_subjects = sorted(avg_subject_scores.items(), key=lambda x: x[1])
        weak_areas = [subject for subject, score in sorted_subjects[:2]] if len(sorted_subjects) >= 2 else []
        strong_areas = [subject for subject, score in sorted_subjects[-2:]] if len(sorted_subjects) >= 2 else []
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(avg_subject_scores, weak_areas)
        
        # Calculate percentile (simplified - would need cohort data for accurate calculation)
        percentile = 75.0  # Placeholder
        
        # Time analysis
        total_time = sum(record.get('time_taken', 0) for record in recent_data)
        avg_time_per_test = total_time / len(recent_data) if recent_data else 0
        
        # Accuracy by topic (would need topic-level data)
        accuracy_by_topic = {}  # Placeholder
        
        return PerformanceMetrics(
            student_id=student_id,
            test_id=recent_data[-1].get('test_id', '') if recent_data else '',
            total_marks=total_marks,
            max_marks=max_mands,
            percentage=overall_percentage,
            rank=None,  # Would need cohort data
            percentile=percentile,
            subject_scores=avg_subject_scores,
            difficulty_scores=difficulty_scores,
            time_taken=avg_time_per_test,
            accuracy_by_topic=accuracy_by_topic,
            weak_areas=weak_areas,
            strong_areas=strong_areas,
            improvement_suggestions=suggestions
        )
    
    def analyze_cohort_performance(self, filters: Dict = None) -> Dict:
        """Analyze performance across a cohort"""
        cohort_data = self.data_aggregator.get_cohort_data(filters)
        
        if not cohort_data:
            return {'error': 'No data found for specified cohort'}
        
        # Overall statistics
        percentages = [record.get('percentage', 0) for record in cohort_data]
        
        analysis = {
            'cohort_size': len(cohort_data),
            'overall_stats': {
                'mean': np.mean(percentages),
                'median': np.median(percentages),
                'std': np.std(percentages),
                'min': np.min(percentages),
                'max': np.max(percentages),
                'q25': np.percentile(percentages, 25),
                'q75': np.percentile(percentages, 75)
            },
            'distribution': self._create_percentage_distribution(percentages),
            'subject_performance': self._analyze_subject_performance(cohort_data),
            'temporal_trends': self._analyze_temporal_trends(cohort_data),
            'difficulty_analysis': self._analyze_difficulty_performance(cohort_data)
        }
        
        return analysis
    
    def _generate_improvement_suggestions(self, subject_scores: Dict[str, float], 
                                        weak_areas: List[str]) -> List[str]:
        """Generate personalized improvement suggestions"""
        suggestions = []
        
        for subject in weak_areas:
            score = subject_scores.get(subject, 0)
            
            if subject == 'mathematics' or subject == 'quantitative_aptitude':
                if score < 50:
                    suggestions.append("Focus on basic arithmetic and number system concepts")
                    suggestions.append("Practice more calculation problems daily")
                elif score < 70:
                    suggestions.append("Strengthen advanced mathematics topics like algebra and geometry")
                    suggestions.append("Time management is crucial for quantitative sections")
            
            elif subject == 'english':
                if score < 50:
                    suggestions.append("Increase daily reading of English newspapers and magazines")
                    suggestions.append("Focus on grammar rules and practice error detection")
                elif score < 70:
                    suggestions.append("Enhance vocabulary through word games and flashcards")
                    suggestions.append("Practice comprehension passages regularly")
            
            elif subject == 'reasoning':
                suggestions.append("Practice logical reasoning puzzles daily")
                suggestions.append("Learn pattern recognition techniques")
            
            elif subject == 'general_awareness':
                suggestions.append("Stay updated with current affairs through daily news reading")
                suggestions.append("Use spaced repetition for static GK facts")
            
            else:
                suggestions.append(f"Focus more practice on {subject} topics")
                suggestions.append("Identify specific weak subtopics within this subject")
        
        return suggestions
    
    def _create_percentage_distribution(self, percentages: List[float]) -> Dict:
        """Create percentage distribution buckets"""
        buckets = {
            '0-30%': 0, '31-50%': 0, '51-60%': 0, 
            '61-70%': 0, '71-80%': 0, '81-90%': 0, '91-100%': 0
        }
        
        for percentage in percentages:
            if percentage <= 30:
                buckets['0-30%'] += 1
            elif percentage <= 50:
                buckets['31-50%'] += 1
            elif percentage <= 60:
                buckets['51-60%'] += 1
            elif percentage <= 70:
                buckets['61-70%'] += 1
            elif percentage <= 80:
                buckets['71-80%'] += 1
            elif percentage <= 90:
                buckets['81-90%'] += 1
            else:
                buckets['91-100%'] += 1
        
        return buckets
    
    def _analyze_subject_performance(self, cohort_data: List[Dict]) -> Dict:
        """Analyze performance by subject"""
        subject_stats = defaultdict(list)
        
        for record in cohort_data:
            for subject, score in record.get('subject_scores', {}).items():
                subject_stats[subject].append(score)
        
        return {
            subject: {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
            for subject, scores in subject_stats.items()
        }
    
    def _analyze_temporal_trends(self, cohort_data: List[Dict]) -> Dict:
        """Analyze performance trends over time"""
        # Group by date and calculate average performance
        daily_performance = defaultdict(list)
        
        for record in cohort_data:
            test_date = datetime.fromisoformat(record.get('test_date', ''))
            date_str = test_date.strftime('%Y-%m-%d')
            daily_performance[date_str].append(record.get('percentage', 0))
        
        # Calculate daily averages
        daily_averages = {
            date: np.mean(scores) 
            for date, scores in daily_performance.items()
        }
        
        # Sort by date
        sorted_trends = dict(sorted(daily_averages.items()))
        
        return {
            'daily_averages': sorted_trends,
            'trend_direction': self._calculate_trend_direction(list(sorted_trends.values()))
        }
    
    def _analyze_difficulty_performance(self, cohort_data: List[Dict]) -> Dict:
        """Analyze performance by question difficulty"""
        # This would require difficulty information in the data
        # Placeholder implementation
        return {
            'easy': {'mean': 75.0, 'count': 100},
            'medium': {'mean': 65.0, 'count': 150},
            'hard': {'mean': 55.0, 'count': 50}
        }
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if abs(slope) < 0.1:  # Very small slope
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"

class PredictiveAnalyzer:
    """Provides predictive analytics and forecasting"""
    
    def __init__(self, data_aggregator: DataAggregator):
        self.data_aggregator = data_aggregator
        self.models = {}
    
    def predict_next_test_score(self, student_id: str, exam_code: str) -> Dict:
        """Predict student's next test score based on historical performance"""
        student_data = self.data_aggregator.get_student_data(student_id)
        exam_data = [r for r in student_data if r.get('exam_code') == exam_code]
        
        if len(exam_data) < 3:
            return {'prediction': 'insufficient_data', 'confidence': 0.0}
        
        # Extract features
        percentages = [r.get('percentage', 0) for r in exam_data]
        
        # Simple trend-based prediction
        if len(percentages) >= 5:
            # Use recent trend
            recent_scores = percentages[-3:]
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            predicted_score = recent_scores[-1] + trend
        else:
            # Use average with small improvement factor
            predicted_score = np.mean(percentages) + 2.0
        
        # Add confidence interval
        recent_std = np.std(percentages[-3:]) if len(percentages) >= 3 else 10
        confidence_interval = 1.96 * recent_std  # 95% confidence interval
        
        return {
            'predicted_score': max(0, min(100, predicted_score)),
            'confidence_interval': confidence_interval,
            'confidence_level': 0.75 if len(exam_data) >= 5 else 0.5,
            'trend_direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
        }
    
    def predict_exam_readiness(self, student_id: str, exam_code: str, 
                             target_score: float) -> Dict:
        """Predict if student is ready for a specific exam"""
        student_data = self.data_aggregator.get_student_data(student_id)
        exam_data = [r for r in student_data if r.get('exam_code') == exam_code]
        
        if not exam_data:
            return {'readiness': 'no_data', 'recommendation': 'Take practice tests first'}
        
        recent_performance = [r.get('percentage', 0) for r in exam_data[-5:]]
        avg_score = np.mean(recent_performance)
        consistency = 1 - (np.std(recent_performance) / 100)  # Consistency score
        
        # Readiness calculation
        score_factor = avg_score / target_score if target_score > 0 else 1
        consistency_factor = consistency
        
        readiness_score = (score_factor + consistency_factor) / 2
        
        if readiness_score >= 0.9:
            readiness = 'ready'
            recommendation = 'You are well prepared. Take full-length mock tests.'
        elif readiness_score >= 0.75:
            readiness = 'almost_ready'
            recommendation = 'You are close. Focus on weak areas and take more practice tests.'
        elif readiness_score >= 0.6:
            readiness = 'needs_work'
            recommendation = 'Focus on fundamentals and take more targeted practice.'
        else:
            readiness = 'not_ready'
            recommendation = 'Start with basic concepts and build strong foundation.'
        
        return {
            'readiness': readiness,
            'readiness_score': readiness_score,
            'recommendation': recommendation,
            'predicted_days_to_readiness': max(0, int((target_score - avg_score) / 2))
        }
    
    def cluster_students(self, features: List[str] = None) -> Dict:
        """Cluster students based on performance patterns"""
        # Get all student data
        all_data = self.data_aggregator.performance_data
        
        if not all_data:
            return {'error': 'No data available for clustering'}
        
        # Prepare features
        if not features:
            features = ['percentage', 'total_marks', 'time_taken']
        
        # Create feature matrix
        student_features = defaultdict(list)
        for record in all_data:
            student_id = record.get('student_id', 'unknown')
            for feature in features:
                student_features[student_id].append(record.get(feature, 0))
        
        # Average features per student
        feature_matrix = []
        student_ids = []
        for student_id, values in student_features.items():
            feature_matrix.append([np.mean(values)])
            student_ids.append(student_id)
        
        if len(feature_matrix) < 3:
            return {'error': 'Insufficient data for clustering'}
        
        feature_matrix = np.array(feature_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        n_clusters = min(5, len(feature_matrix) // 2)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_students = [student_ids[j] for j, cluster in enumerate(clusters) if cluster == i]
            cluster_scores = [feature_matrix[j][0] for j, cluster in enumerate(clusters) if cluster == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'students': cluster_students,
                'count': len(cluster_students),
                'avg_score': np.mean(cluster_scores),
                'characteristics': self._analyze_cluster_characteristics(cluster_students)
            }
        
        return {
            'clusters': cluster_analysis,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'total_students': len(student_ids)
        }
    
    def _analyze_cluster_characteristics(self, student_ids: List[str]) -> Dict:
        """Analyze characteristics of a student cluster"""
        cluster_data = []
        for student_id in student_ids:
            student_records = self.data_aggregator.get_student_data(student_id)
            cluster_data.extend(student_records)
        
        if not cluster_data:
            return {}
        
        percentages = [r.get('percentage', 0) for r in cluster_data]
        
        return {
            'avg_performance': np.mean(percentages),
            'performance_range': f"{np.min(percentages):.1f} - {np.max(percentages):.1f}",
            'most_common_exam': Counter(r.get('exam_code', '') for r in cluster_data).most_common(1)[0][0]
        }

class VisualizationEngine:
    """Creates visualizations for analytics"""
    
    def __init__(self, output_dir: str = "analytics/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_student_progress(self, student_id: str, performance_metrics: PerformanceMetrics,
                            save_path: bool = True) -> str:
        """Plot individual student progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Student {student_id} Performance Analysis', fontsize=16)
        
        # Subject performance radar chart
        if performance_metrics.subject_scores:
            subjects = list(performance_metrics.subject_scores.keys())
            scores = list(performance_metrics.subject_scores.values())
            
            # Convert to radar chart data
            angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
            scores += scores[:1]  # Complete the circle
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 2, 1, projection='polar')
            ax_radar.plot(angles, scores, 'o-', linewidth=2, label='Performance')
            ax_radar.fill(angles, scores, alpha=0.25)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(subjects)
            ax_radar.set_ylim(0, 100)
            ax_radar.set_title('Subject-wise Performance')
            ax_radar.grid(True)
        
        # Overall performance gauge
        ax_gauge = plt.subplot(2, 2, 2)
        performance = performance_metrics.percentage
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        color = colors[int(performance // 20)]
        
        ax_gauge.pie([performance, 100-performance], colors=[color, 'lightgray'], 
                    startangle=90, counterclock=False)
        ax_gauge.text(0, 0, f'{performance:.1f}%', ha='center', va='center', 
                     fontsize=20, fontweight='bold')
        ax_gauge.set_title('Overall Performance')
        
        # Weak vs Strong areas
        ax_areas = plt.subplot(2, 2, 3)
        areas_data = {
            'Strong Areas': len(performance_metrics.strong_areas),
            'Weak Areas': len(performance_metrics.weak_areas),
            'Neutral': max(0, len(performance_metrics.subject_scores) - 
                          len(performance_metrics.strong_areas) - 
                          len(performance_metrics.weak_areas))
        }
        
        ax_areas.bar(areas_data.keys(), areas_data.values(), 
                    color=['green', 'red', 'gray'])
        ax_areas.set_title('Performance Distribution')
        ax_areas.set_ylabel('Number of Subjects')
        
        # Time analysis (placeholder)
        ax_time = plt.subplot(2, 2, 4)
        ax_time.text(0.5, 0.5, f'Average Time per Test:\n{performance_metrics.time_taken:.1f} minutes',
                    ha='center', va='center', fontsize=14)
        ax_time.set_title('Time Management')
        ax_time.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_file = self.output_dir / f'{student_id}_progress.png'
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_file)
        else:
            plt.show()
            return ""
    
    def plot_cohort_analysis(self, cohort_analysis: Dict, save_path: bool = True) -> str:
        """Plot cohort analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cohort Performance Analysis', fontsize=16)
        
        # Performance distribution histogram
        ax_hist = axes[0, 0]
        # This would need actual percentage data
        ax_hist.hist([60, 65, 70, 75, 80, 85], bins=10, alpha=0.7, color='skyblue')
        ax_hist.set_title('Performance Distribution')
        ax_hist.set_xlabel('Score Range (%)')
        ax_hist.set_ylabel('Frequency')
        
        # Subject performance comparison
        ax_subject = axes[0, 1]
        subject_stats = cohort_analysis.get('subject_performance', {})
        if subject_stats:
            subjects = list(subject_stats.keys())[:5]  # Top 5 subjects
            means = [subject_stats[s]['mean'] for s in subjects]
            stds = [subject_stats[s]['std'] for s in subjects]
            
            ax_subject.bar(subjects, means, yerr=stds, capsize=5, color='lightgreen')
            ax_subject.set_title('Average Subject Performance')
            ax_subject.set_ylabel('Average Score')
            ax_subject.tick_params(axis='x', rotation=45)
        
        # Temporal trends
        ax_trend = axes[1, 0]
        temporal_data = cohort_analysis.get('temporal_trends', {})
        daily_averages = temporal_data.get('daily_averages', {})
        if daily_averages:
            dates = list(daily_averages.keys())[-10:]  # Last 10 days
            scores = list(daily_averages.values())[-10:]
            
            ax_trend.plot(dates, scores, marker='o')
            ax_trend.set_title('Performance Trends')
            ax_trend.set_xlabel('Date')
            ax_trend.set_ylabel('Average Score')
            ax_trend.tick_params(axis='x', rotation=45)
        
        # Performance buckets
        ax_buckets = axes[1, 1]
        distribution = cohort_analysis.get('distribution', {})
        if distribution:
            buckets = list(distribution.keys())
            counts = list(distribution.values())
            
            ax_buckets.bar(buckets, counts, color='coral')
            ax_buckets.set_title('Score Distribution')
            ax_buckets.set_ylabel('Number of Students')
            ax_buckets.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            save_file = self.output_dir / 'cohort_analysis.png'
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_file)
        else:
            plt.show()
            return ""

class AnalyticsEngine:
    """Main analytics engine orchestrator"""
    
    def __init__(self, data_path: str = "data/performance"):
        self.data_aggregator = DataAggregator(data_path)
        self.performance_analyzer = PerformanceAnalyzer(self.data_aggregator)
        self.predictive_analyzer = PredictiveAnalyzer(self.data_aggregator)
        self.visualization_engine = VisualizationEngine()
    
    def generate_student_report(self, student_id: str, days_back: int = 30) -> AnalyticsReport:
        """Generate comprehensive student analytics report"""
        logger.info(f"Generating analytics report for student {student_id}")
        
        # Get performance metrics
        performance_metrics = self.performance_analyzer.analyze_individual_performance(
            student_id, days_back
        )
        
        # Generate predictions
        predictions = {
            'next_test_score': self.predictive_analyzer.predict_next_test_score(
                student_id, performance_metrics.test_id.split('_')[0] if '_' in performance_metrics.test_id else ''
            ),
            'exam_readiness': self.predictive_analyzer.predict_exam_readiness(
                student_id, 'ssc_cgl', 75.0  # Default target
            )
        }
        
        # Generate recommendations
        recommendations = {
            'immediate_focus': performance_metrics.weak_areas,
            'study_suggestions': performance_metrics.improvement_suggestions,
            'next_steps': self._generate_next_steps(performance_metrics, predictions)
        }
        
        # Create analytics report
        time_period = (
            datetime.now() - timedelta(days=days_back),
            datetime.now()
        )
        
        report = AnalyticsReport(
            report_id=f"report_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            student_id=student_id,
            generated_at=datetime.now(),
            time_period=time_period,
            overall_performance={
                'total_marks': performance_metrics.total_marks,
                'max_marks': performance_metrics.max_marks,
                'percentage': performance_metrics.percentage,
                'percentile': performance_metrics.percentile
            },
            subject_analysis=performance_metrics.subject_scores,
            difficulty_analysis=performance_metrics.difficulty_scores,
            temporal_analysis={},  # Would need historical data
            comparative_analysis={},  # Would need cohort comparison
            predictions=predictions,
            recommendations=recommendations
        )
        
        # Generate visualizations
        try:
            progress_chart = self.visualization_engine.plot_student_progress(
                student_id, performance_metrics
            )
            report.metadata = {'progress_chart': progress_chart}
        except Exception as e:
            logger.warning(f"Could not generate visualization: {str(e)}")
        
        return report
    
    def generate_cohort_report(self, filters: Dict = None) -> Dict:
        """Generate cohort analytics report"""
        logger.info("Generating cohort analytics report")
        
        cohort_analysis = self.performance_analyzer.analyze_cohort_performance(filters)
        
        # Add clustering analysis
        cluster_analysis = self.predictive_analyzer.cluster_students()
        cohort_analysis['student_clusters'] = cluster_analysis
        
        # Generate visualizations
        try:
            cohort_chart = self.visualization_engine.plot_cohort_analysis(cohort_analysis)
            cohort_analysis['visualization'] = cohort_chart
        except Exception as e:
            logger.warning(f"Could not generate cohort visualization: {str(e)}")
        
        return cohort_analysis
    
    def _generate_next_steps(self, performance_metrics: PerformanceMetrics, 
                           predictions: Dict) -> List[str]:
        """Generate next steps for student improvement"""
        next_steps = []
        
        # Based on performance
        if performance_metrics.percentage < 50:
            next_steps.append("Focus on foundational concepts before attempting difficult questions")
            next_steps.append("Take more practice tests to identify specific weak areas")
        
        # Based on predictions
        exam_readiness = predictions.get('exam_readiness', {})
        if exam_readiness.get('readiness') == 'not_ready':
            next_steps.append("Increase study time and seek additional help in weak subjects")
        elif exam_readiness.get('readiness') == 'needs_work':
            next_steps.append("Targeted practice in identified weak areas")
            next_steps.append("Take subject-specific mock tests")
        
        # Time management
        if performance_metrics.time_taken > 120:  # Assuming 2 hours
            next_steps.append("Improve time management through timed practice sessions")
        
        return next_steps
    
    def save_report(self, report: AnalyticsReport, output_dir: str = "analytics/reports"):
        """Save analytics report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert to serializable format
        report_data = {
            'report_id': report.report_id,
            'student_id': report.student_id,
            'generated_at': report.generated_at.isoformat(),
            'time_period': {
                'start': report.time_period[0].isoformat(),
                'end': report.time_period[1].isoformat()
            },
            'overall_performance': report.overall_performance,
            'subject_analysis': report.subject_analysis,
            'difficulty_analysis': report.difficulty_analysis,
            'temporal_analysis': report.temporal_analysis,
            'comparative_analysis': report.comparative_analysis,
            'predictions': report.predictions,
            'recommendations': report.recommendations,
            'metadata': getattr(report, 'metadata', {})
        }
        
        file_path = output_path / f"{report.report_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved analytics report to {file_path}")
        return str(file_path)

# Usage example
if __name__ == "__main__":
    # Initialize analytics engine
    analytics = AnalyticsEngine()
    
    # Generate student report
    try:
        student_report = analytics.generate_student_report("student_001")
        print(f"Generated report: {student_report.report_id}")
        print(f"Overall performance: {student_report.overall_performance['percentage']:.1f}%")
        
        # Save report
        report_file = analytics.save_report(student_report)
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error generating student report: {e}")
    
    # Generate cohort report
    try:
        cohort_report = analytics.generate_cohort_report()
        print(f"Cohort size: {cohort_report.get('cohort_size', 'N/A')}")
        print(f"Average performance: {cohort_report.get('overall_stats', {}).get('mean', 'N/A'):.1f}%")
        
    except Exception as e:
        print(f"Error generating cohort report: {e}")