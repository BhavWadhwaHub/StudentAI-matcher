"""
Comprehensive Evaluation Script for Student Matching Model

This script provides complete evaluation of the matching model including:
- Basic metrics (similarity scores, match counts)
- Coverage metrics (how well students are matched)
- Quality metrics (match quality distribution)
- Diversity metrics (diversity enforcement analysis)
- Clustering metrics (K-means clustering quality)
- Stability metrics (model consistency)
- Diversity bonus analysis (with vs without diversity bonuses)

Supports loading data from:
1. Node.js API (default - same as Flask app)
2. Google Sheets directly (using gspread)
3. Local CSV file (fallback)

All results are saved to a single output directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys
import requests

# Add app_backend to path for imports
app_backend_path = os.path.join(os.path.dirname(__file__), '..', 'app_backend')
if os.path.exists(app_backend_path):
    sys.path.insert(0, app_backend_path)
else:
    # Try current directory if running from app_backend
    sys.path.insert(0, os.path.dirname(__file__))

from matcher import (
    generate_matches, calculate_weighted_similarity, preprocess_data, run_kmeans,
    calculate_diversity_bonus, is_diverse_match
)
from config import (
    TOP_MATCHES_COUNT, MATCHING_ALGORITHM, ENABLE_DIVERSITY_ENFORCEMENT,
    DIVERSITY_ATTRIBUTES, MATCHING_WEIGHTS, SIMILARITY_THRESHOLDS,
    DIVERSITY_BONUS, MAX_DIVERSITY_BONUS
)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data_from_api(api_url="http://localhost:3001/api/students"):
    """
    Load student data from Node.js API (which fetches from Google Sheets).
    
    Args:
        api_url: URL of the Node.js API endpoint
        
    Returns:
        DataFrame with student data
    """
    try:
        print(f"Fetching data from Node.js API: {api_url}")
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        students = response.json()
        print(f"[OK] Loaded {len(students)} students from Node.js API")
        
        if not students:
            print("[WARN] Warning: API returned empty list")
            return None
        
        df = pd.DataFrame(students)
        return df
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Connection error: Node.js server not running on {api_url}")
        print("   Make sure the Node.js server is running (npm start)")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading from API: {e}")
        return None


def load_data_from_google_sheets(spreadsheet_id=None, sheet_name='Students', credentials_path=None):
    """
    Load student data directly from Google Sheets using gspread.
    
    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        sheet_name: Name of the sheet tab
        credentials_path: Path to Google service account credentials JSON file
        
    Returns:
        DataFrame with student data
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        if not spreadsheet_id:
            print("[ERROR] Error: spreadsheet_id is required for Google Sheets access")
            return None
        
        if credentials_path and os.path.exists(credentials_path):
            creds = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
        else:
            # Try to use environment variable or default location
            default_path = os.path.join(os.path.dirname(__file__), '..', 'app_backend', 'google_credentials.json')
            if os.path.exists(default_path):
                creds = Credentials.from_service_account_file(
                    default_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
                )
            else:
                print("[ERROR] Error: Google credentials file not found")
                print("   Please provide credentials_path or place credentials at:")
                print(f"   {default_path}")
                return None
        
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all records
        records = worksheet.get_all_records()
        print(f"[OK] Loaded {len(records)} students from Google Sheets")
        
        df = pd.DataFrame(records)
        return df
    except ImportError:
        print("[ERROR] Error: gspread not installed")
        print("   Install with: pip install gspread google-auth")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading from Google Sheets: {e}")
        return None


def load_data_from_csv(csv_path='app_backend/data/students.csv'):
    """
    Load student data from local CSV file (fallback).
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with student data
    """
    try:
        if not os.path.exists(csv_path):
            print(f"[ERROR] Error: CSV file not found at {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        # Normalize column names to lowercase (handle both 'Name' and 'name')
        column_mapping = {}
        for col in df.columns:
            if col == 'Name':
                column_mapping[col] = 'name'
            elif col == 'Email':
                column_mapping[col] = 'email'
            elif col == 'Major':
                column_mapping[col] = 'major'
            elif col == 'Year':
                column_mapping[col] = 'year'
            elif col == 'Language':
                column_mapping[col] = 'language'
            elif col == 'Country':
                column_mapping[col] = 'country'
            elif col == 'Personality':
                column_mapping[col] = 'personality'
            elif col == 'Study Style':
                column_mapping[col] = 'studyStyle'
            elif col == 'Cuisine':
                column_mapping[col] = 'cuisine'
            elif col == 'Interests':
                column_mapping[col] = 'interests'
            elif col == 'Movies':
                column_mapping[col] = 'movies'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        print(f"[OK] Loaded {len(df)} students from CSV file")
        return df
    except Exception as e:
        print(f"[ERROR] Error loading from CSV: {e}")
        return None


# ============================================================================
# DIVERSITY ANALYSIS HELPER FUNCTIONS
# ============================================================================

def calculate_matches_without_diversity(df):
    """Calculate matches without diversity bonuses for comparison."""
    matches_no_diversity = {}
    
    for idx, student in df.iterrows():
        student_name = str(student['name']).strip()
        student_data = student.to_dict()
        
        match_scores = []
        
        for idx2, other_student in df.iterrows():
            if idx == idx2:
                continue
            
            other_name = str(other_student['name']).strip()
            other_data = other_student.to_dict()
            
            # Calculate similarity WITHOUT diversity bonus
            similarity = calculate_weighted_similarity(
                student_data, other_data, include_diversity_bonus=False
            )
            
            match_scores.append((other_name, similarity))
        
        # Sort by score and take top matches
        match_scores.sort(key=lambda x: x[1], reverse=True)
        matches_no_diversity[student_name] = match_scores[:TOP_MATCHES_COUNT]
    
    return matches_no_diversity


def calculate_matches_with_diversity(df):
    """Calculate matches with diversity bonuses."""
    matches_with_diversity = {}
    
    for idx, student in df.iterrows():
        student_name = str(student['name']).strip()
        student_data = student.to_dict()
        
        match_scores = []
        
        for idx2, other_student in df.iterrows():
            if idx == idx2:
                continue
            
            other_name = str(other_student['name']).strip()
            other_data = other_student.to_dict()
            
            # Calculate similarity WITH diversity bonus
            similarity = calculate_weighted_similarity(
                student_data, other_data, include_diversity_bonus=True
            )
            
            # Check if diverse
            is_diverse = is_diverse_match(student_data, other_data)
            
            match_scores.append((other_name, similarity, is_diverse))
        
        # Sort by score and take top matches
        match_scores.sort(key=lambda x: x[1], reverse=True)
        matches_with_diversity[student_name] = match_scores[:TOP_MATCHES_COUNT]
    
    return matches_with_diversity


# ============================================================================
# COMPREHENSIVE MODEL EVALUATOR CLASS
# ============================================================================

class ComprehensiveModelEvaluator:
    """Comprehensive evaluator for student matching model with all metrics."""
    
    def __init__(self, df, output_dir='evaluation_results'):
        """
        Initialize evaluator.
        
        Args:
            df: DataFrame with student data
            output_dir: Directory to save evaluation results and plots
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.matches = None
        self.metrics = {}
        self.diversity_bonus_stats = None
        self.diversity_comparison = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate(self, include_diversity_analysis=True):
        """
        Run all evaluations and generate reports.
        
        Args:
            include_diversity_analysis: If True, includes detailed diversity bonus analysis
        """
        print("=" * 60)
        print("COMPREHENSIVE STUDENT MATCHING MODEL EVALUATION")
        print("=" * 60)
        
        # Step 1: Find optimal k first (if using clustering)
        self.optimal_k_used = None
        optimal_k = None
        original_find_best_k = None
        original_kmeans_n_clusters = None
        if MATCHING_ALGORITHM in ['kmeans', 'hybrid']:
            print("\n1. Finding optimal k value...")
            optimal_k = self._find_optimal_k_for_evaluation()
            if optimal_k:
                print(f"   Using optimal k={optimal_k} for evaluation")
                self.optimal_k_used = optimal_k
                # Temporarily patch find_best_k to return optimal k (for kmeans algorithm)
                from matcher import find_best_k
                original_find_best_k = find_best_k
                
                # Create a patched version that returns optimal k
                def patched_find_best_k(df, min_k=2, max_k=12):
                    return optimal_k
                
                # Monkey patch the function
                import matcher
                matcher.find_best_k = patched_find_best_k
                
                # Also set KMEANS_N_CLUSTERS for hybrid algorithm
                import config
                original_kmeans_n_clusters = config.KMEANS_N_CLUSTERS
                config.KMEANS_N_CLUSTERS = optimal_k
            else:
                print("   Could not determine optimal k, using default")
        else:
            print("\n1. Skipping k optimization (not using clustering algorithm)")
        
        # Step 2: Generate matches with optimal k
        print("\n2. Generating matches...")
        self.matches = generate_matches(self.df)
        
        # Restore original function and config if we patched them
        if original_find_best_k:
            import matcher
            matcher.find_best_k = original_find_best_k
        if original_kmeans_n_clusters is not None:
            import config
            config.KMEANS_N_CLUSTERS = original_kmeans_n_clusters
        
        if not self.matches:
            print("[WARN] No matches generated. Check your data and configuration.")
            return
        
        # Step 3: Calculate basic metrics
        print("\n3. Calculating metrics...")
        self._calculate_basic_metrics()
        self._calculate_coverage_metrics()
        self._calculate_quality_metrics()
        self._calculate_diversity_metrics()
        self._calculate_cluster_metrics()
        self._analyze_optimal_k()  # Still analyze for comparison/visualization
        self._calculate_stability_metrics()
        
        # Step 4: Diversity bonus analysis (if enabled)
        if include_diversity_analysis and ENABLE_DIVERSITY_ENFORCEMENT:
            print("\n4. Analyzing diversity bonus impact...")
            self._analyze_diversity_bonus_impact()
        
        # Step 5: Generate visualizations
        print("\n5. Generating visualizations...")
        self._plot_similarity_distribution()
        self._plot_match_quality_distribution()
        self._plot_coverage_metrics()
        self._plot_diversity_metrics()
        self._plot_cluster_analysis()
        self._plot_optimal_k_combined()
        self._plot_feature_importance()
        self._plot_match_network()
        
        # Diversity bonus visualizations
        if include_diversity_analysis and ENABLE_DIVERSITY_ENFORCEMENT and self.diversity_bonus_stats:
            self._plot_diversity_bonus_analysis()
        
        # Step 6: Generate report
        print("\n6. Generating evaluation report...")
        self._generate_report()
        
        print(f"\n[OK] Evaluation complete! Results saved to '{self.output_dir}/'")
    
    def _calculate_basic_metrics(self):
        """Calculate basic matching metrics."""
        all_scores = []
        match_counts = []
        
        for student_name, student_matches in self.matches.items():
            if student_matches:
                scores = [match['similarity_score'] for match in student_matches]
                all_scores.extend(scores)
                match_counts.append(len(student_matches))
        
        self.metrics['basic'] = {
            'total_students': len(self.df),
            'students_with_matches': len(self.matches),
            'total_matches': sum(len(matches) for matches in self.matches.values()),
            'avg_matches_per_student': np.mean(match_counts) if match_counts else 0,
            'avg_similarity_score': np.mean(all_scores) if all_scores else 0,
            'median_similarity_score': np.median(all_scores) if all_scores else 0,
            'std_similarity_score': np.std(all_scores) if all_scores else 0,
            'min_similarity_score': np.min(all_scores) if all_scores else 0,
            'max_similarity_score': np.max(all_scores) if all_scores else 0,
        }
    
    def _calculate_coverage_metrics(self):
        """Calculate coverage metrics (how well students are matched)."""
        students_with_matches = len(self.matches)
        total_students = len(self.df)
        coverage = (students_with_matches / total_students * 100) if total_students > 0 else 0
        
        # Calculate reciprocal matches (A matches B and B matches A)
        reciprocal_count = 0
        total_pairs = 0
        
        for student1, matches1 in self.matches.items():
            for match in matches1:
                match_name = match['name']
                if match_name in self.matches:
                    matches2 = self.matches[match_name]
                    if any(m['name'] == student1 for m in matches2):
                        reciprocal_count += 1
                    total_pairs += 1
        
        reciprocal_rate = (reciprocal_count / total_pairs * 100) if total_pairs > 0 else 0
        
        self.metrics['coverage'] = {
            'coverage_percentage': coverage,
            'students_with_matches': students_with_matches,
            'total_students': total_students,
            'reciprocal_match_rate': reciprocal_rate,
            'reciprocal_matches': reciprocal_count,
            'total_match_pairs': total_pairs
        }
    
    def _calculate_quality_metrics(self):
        """Calculate match quality metrics."""
        quality_distribution = {
            'excellent': 0,
            'very_good': 0,
            'good': 0,
            'decent': 0
        }
        
        all_scores = []
        for student_name, student_matches in self.matches.items():
            for match in student_matches:
                score = match['similarity_score']
                all_scores.append(score)
                
                if score >= SIMILARITY_THRESHOLDS['excellent']:
                    quality_distribution['excellent'] += 1
                elif score >= SIMILARITY_THRESHOLDS['very_good']:
                    quality_distribution['very_good'] += 1
                elif score >= SIMILARITY_THRESHOLDS['good']:
                    quality_distribution['good'] += 1
                else:
                    quality_distribution['decent'] += 1
        
        total_matches = len(all_scores)
        quality_percentages = {
            k: (v / total_matches * 100) if total_matches > 0 else 0
            for k, v in quality_distribution.items()
        }
        
        self.metrics['quality'] = {
            'quality_distribution': quality_distribution,
            'quality_percentages': quality_percentages,
            'total_matches': total_matches,
            'score_distribution': {
                'q25': np.percentile(all_scores, 25) if all_scores else 0,
                'q50': np.percentile(all_scores, 50) if all_scores else 0,
                'q75': np.percentile(all_scores, 75) if all_scores else 0,
                'q90': np.percentile(all_scores, 90) if all_scores else 0,
            }
        }
    
    def _calculate_diversity_metrics(self):
        """Calculate diversity metrics if diversity enforcement is enabled."""
        if not ENABLE_DIVERSITY_ENFORCEMENT:
            self.metrics['diversity'] = {'enabled': False}
            return
        
        diverse_matches = 0
        total_matches = 0
        diversity_by_attribute = defaultdict(int)
        
        for student_name, student_matches in self.matches.items():
            # Get student data
            student_filter = self.df[self.df['name'].astype(str).str.strip() == student_name]
            if len(student_filter) == 0:
                continue
            student_data = student_filter.iloc[0].to_dict()
            
            for match in student_matches:
                match_name = match['name']
                match_filter = self.df[self.df['name'].astype(str).str.strip() == match_name]
                if len(match_filter) == 0:
                    continue
                match_data = match_filter.iloc[0].to_dict()
                
                total_matches += 1
                is_diverse = False
                
                # Check diversity attributes
                for attr in DIVERSITY_ATTRIBUTES:
                    if attr in student_data and attr in match_data:
                        val1 = student_data[attr]
                        val2 = match_data[attr]
                        if val1 != val2 and val1 and val2:
                            is_diverse = True
                            diversity_by_attribute[attr] += 1
                
                if is_diverse:
                    diverse_matches += 1
        
        diversity_rate = (diverse_matches / total_matches * 100) if total_matches > 0 else 0
        
        self.metrics['diversity'] = {
            'enabled': True,
            'diverse_matches': diverse_matches,
            'total_matches': total_matches,
            'diversity_rate': diversity_rate,
            'diversity_by_attribute': dict(diversity_by_attribute)
        }
    
    def _calculate_cluster_metrics(self):
        """Calculate clustering quality metrics if using K-means."""
        if MATCHING_ALGORITHM not in ['kmeans', 'hybrid']:
            self.metrics['clustering'] = {'algorithm': MATCHING_ALGORITHM, 'clustering_used': False}
            return
        
        try:
            preprocessed_df, _ = preprocess_data(self.df)
            if preprocessed_df is None or len(preprocessed_df) == 0:
                self.metrics['clustering'] = {'error': 'Could not preprocess data'}
                return
            
            labels, kmeans_model = run_kmeans(preprocessed_df, n_clusters=None)
            if labels is None or kmeans_model is None:
                self.metrics['clustering'] = {'error': 'Could not run clustering'}
                return
            
            # Calculate silhouette score
            feature_matrix = preprocessed_df.values
            silhouette = silhouette_score(feature_matrix, labels) if len(set(labels)) > 1 else -1
            
            # Calculate inertia
            inertia = kmeans_model.inertia_
            
            # Cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            self.metrics['clustering'] = {
                'algorithm': MATCHING_ALGORITHM,
                'clustering_used': True,
                'n_clusters': len(unique_labels),
                'silhouette_score': silhouette,
                'inertia': inertia,
                'cluster_sizes': cluster_sizes,
                'avg_cluster_size': np.mean(counts) if len(counts) > 0 else 0,
                'std_cluster_size': np.std(counts) if len(counts) > 0 else 0
            }
        except Exception as e:
            self.metrics['clustering'] = {'error': str(e)}
    
    def _find_optimal_k_for_evaluation(self):
        """
        Find optimal k value to use for evaluation.
        This runs BEFORE generating matches, so we can use the optimal k.
        
        Returns:
            int: Optimal k value, or None if could not determine
        """
        try:
            preprocessed_df, _ = preprocess_data(self.df)
            if preprocessed_df is None or len(preprocessed_df) == 0:
                return None
            
            feature_matrix = preprocessed_df.values
            from config import KMEANS_MIN_CLUSTERS, KMEANS_MAX_CLUSTERS, RANDOM_SEED
            
            min_k = max(2, KMEANS_MIN_CLUSTERS)
            max_k = min(KMEANS_MAX_CLUSTERS, len(feature_matrix) - 1, 20)
            
            k_values = list(range(min_k, max_k + 1))
            silhouette_scores = []
            
            print(f"   Testing k values from {min_k} to {max_k}...")
            for k in k_values:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                    labels = kmeans.fit_predict(feature_matrix)
                    
                    if len(set(labels)) > 1:
                        silhouette = silhouette_score(feature_matrix, labels)
                        silhouette_scores.append(silhouette)
                    else:
                        silhouette_scores.append(-1)
                except Exception:
                    silhouette_scores.append(-1)
            
            # Find best k by silhouette score
            valid_silhouettes = [s for s in silhouette_scores if s >= 0]
            if valid_silhouettes:
                best_k_idx = silhouette_scores.index(max(valid_silhouettes))
                best_k = k_values[best_k_idx]
                best_score = max(valid_silhouettes)
                print(f"   Best k={best_k} with silhouette score={best_score:.3f}")
                return best_k
            else:
                return min_k
                
        except Exception as e:
            print(f"   Error finding optimal k: {e}")
            return None
    
    def _analyze_optimal_k(self):
        """Analyze optimal k values and create combined normalized view."""
        if MATCHING_ALGORITHM not in ['kmeans', 'hybrid']:
            return
        
        try:
            print("   Analyzing optimal k values...")
            preprocessed_df, _ = preprocess_data(self.df)
            if preprocessed_df is None or len(preprocessed_df) == 0:
                return
            
            feature_matrix = preprocessed_df.values
            from config import KMEANS_MIN_CLUSTERS, KMEANS_MAX_CLUSTERS, RANDOM_SEED
            
            min_k = max(2, KMEANS_MIN_CLUSTERS)
            max_k = min(KMEANS_MAX_CLUSTERS, len(feature_matrix) - 1, 20)
            
            k_values = list(range(min_k, max_k + 1))
            inertias = []
            silhouette_scores = []
            balance_scores = []
            
            for k in k_values:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                    labels = kmeans.fit_predict(feature_matrix)
                    
                    inertias.append(kmeans.inertia_)
                    
                    if len(set(labels)) > 1:
                        silhouette_scores.append(silhouette_score(feature_matrix, labels))
                    else:
                        silhouette_scores.append(-1)
                    
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    balance_score = 1.0 / (1.0 + np.std(counts))
                    balance_scores.append(balance_score)
                    
                except Exception:
                    inertias.append(float('inf'))
                    silhouette_scores.append(-1)
                    balance_scores.append(0)
            
            # Find best k by silhouette
            valid_silhouettes = [s for s in silhouette_scores if s >= 0]
            if valid_silhouettes:
                best_k_idx = silhouette_scores.index(max(valid_silhouettes))
                best_k = k_values[best_k_idx]
            else:
                best_k = min_k
            
            # Store for plotting
            self.metrics['k_analysis'] = {
                'k_values': k_values,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'balance_scores': balance_scores,
                'best_k': best_k
            }
            
            print(f"   Best k (by silhouette): {best_k}")
            
        except Exception as e:
            print(f"   Error analyzing optimal k: {e}")
    
    def _calculate_stability_metrics(self):
        """Calculate model stability by running multiple times with different seeds."""
        print("   Calculating stability metrics (this may take a moment)...")
        
        original_matches = self.matches.copy()
        stability_scores = []
        
        # Test with different random seeds
        for seed in [42, 123, 456, 789, 999]:
            try:
                # Temporarily modify random seed in config
                from config import RANDOM_SEED
                import random
                random.seed(seed)
                
                # Generate matches with different seed
                test_matches = generate_matches(self.df)
                
                # Calculate Jaccard similarity between original and test matches
                if test_matches:
                    jaccard_similarity = self._calculate_match_similarity(original_matches, test_matches)
                    stability_scores.append(jaccard_similarity)
            except Exception as e:
                print(f"   Warning: Could not test stability with seed {seed}: {e}")
                continue
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        
        self.metrics['stability'] = {
            'avg_jaccard_similarity': avg_stability,
            'stability_scores': stability_scores,
            'n_tests': len(stability_scores)
        }
    
    def _calculate_match_similarity(self, matches1, matches2):
        """Calculate Jaccard similarity between two match sets."""
        all_students = set(matches1.keys()) | set(matches2.keys())
        if not all_students:
            return 0.0
        
        similarities = []
        for student in all_students:
            matches1_set = set(m['name'] for m in matches1.get(student, []))
            matches2_set = set(m['name'] for m in matches2.get(student, []))
            
            if not matches1_set and not matches2_set:
                similarity = 1.0
            elif not matches1_set or not matches2_set:
                similarity = 0.0
            else:
                intersection = len(matches1_set & matches2_set)
                union = len(matches1_set | matches2_set)
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _analyze_diversity_bonus_impact(self):
        """Analyze diversity bonus impact by comparing with and without bonuses."""
        print("   Calculating matches WITHOUT diversity bonuses...")
        matches_no_diversity = calculate_matches_without_diversity(self.df)
        
        print("   Calculating matches WITH diversity bonuses...")
        matches_with_diversity = calculate_matches_with_diversity(self.df)
        
        print("   Analyzing bonus application...")
        self.diversity_bonus_stats = self._analyze_diversity_bonus_application(matches_with_diversity)
        
        print("   Comparing matches...")
        self.diversity_comparison = self._compare_diversity_matches(matches_no_diversity, matches_with_diversity)
    
    def _analyze_diversity_bonus_application(self, matches_with_diversity):
        """Analyze how often diversity bonuses are applied."""
        bonus_stats = {
            'total_matches': 0,
            'matches_with_bonus': 0,
            'bonus_by_attribute': defaultdict(int),
            'bonus_distribution': [],
            'bonus_impact': []
        }
        
        for student_name, student_matches in matches_with_diversity.items():
            student_filter = self.df[self.df['name'].astype(str).str.strip() == student_name]
            if len(student_filter) == 0:
                continue
            
            student_data = student_filter.iloc[0].to_dict()
            
            for match in student_matches:
                match_name = match[0]
                match_score = match[1]
                is_diverse = match[2] if len(match) > 2 else False
                
                bonus_stats['total_matches'] += 1
                
                if is_diverse:
                    match_filter = self.df[self.df['name'].astype(str).str.strip() == match_name]
                    if len(match_filter) > 0:
                        match_data = match_filter.iloc[0].to_dict()
                        
                        # Calculate bonus
                        bonus = calculate_diversity_bonus(student_data, match_data)
                        
                        if bonus > 0:
                            bonus_stats['matches_with_bonus'] += 1
                            bonus_stats['bonus_distribution'].append(bonus)
                            
                            # Calculate score without bonus for comparison
                            score_without_bonus = calculate_weighted_similarity(
                                student_data, match_data, include_diversity_bonus=False
                            )
                            impact = match_score - score_without_bonus
                            bonus_stats['bonus_impact'].append(impact)
                            
                            # Track which attributes contributed
                            for attr in DIVERSITY_ATTRIBUTES:
                                val1 = student_data.get(attr)
                                val2 = match_data.get(attr)
                                if val1 is not None and val2 is not None:
                                    try:
                                        if pd.isna(val1) or pd.isna(val2):
                                            continue
                                    except:
                                        pass
                                    if val1 != val2 and attr in DIVERSITY_BONUS:
                                        bonus_stats['bonus_by_attribute'][attr] += 1
        
        return bonus_stats
    
    def _compare_diversity_matches(self, matches_no_diversity, matches_with_diversity):
        """Compare matches with and without diversity bonuses."""
        comparison = {
            'same_matches': 0,
            'different_matches': 0,
            'score_differences': [],
            'diversity_improvements': 0,
            'quality_changes': []
        }
        
        for student_name in matches_no_diversity.keys():
            if student_name not in matches_with_diversity:
                continue
            
            matches_no_div = [m[0] for m in matches_no_diversity[student_name]]
            matches_with_div = [m[0] for m in matches_with_diversity[student_name]]
            
            # Check if matches changed
            if set(matches_no_div) == set(matches_with_div):
                comparison['same_matches'] += 1
            else:
                comparison['different_matches'] += 1
            
            # Compare scores
            for i, (match_name, score_no_div) in enumerate(matches_no_diversity[student_name]):
                if i < len(matches_with_diversity[student_name]):
                    match_name_with, score_with_div = matches_with_diversity[student_name][i][:2]
                    if match_name == match_name_with:
                        diff = score_with_div - score_no_div
                        comparison['score_differences'].append(diff)
            
            # Check diversity improvement
            student_filter = self.df[self.df['name'].astype(str).str.strip() == student_name]
            if len(student_filter) > 0:
                student_data = student_filter.iloc[0].to_dict()
                
                diverse_no_div = sum(1 for m in matches_no_div 
                                    if is_diverse_match(student_data, 
                                      self.df[self.df['name'].astype(str).str.strip() == m].iloc[0].to_dict() 
                                      if len(self.df[self.df['name'].astype(str).str.strip() == m]) > 0 
                                      else {})
                                    )
                
                diverse_with_div = 0
                for match_info in matches_with_diversity[student_name]:
                    if len(match_info) > 2 and match_info[2]:
                        diverse_with_div += 1
                
                if diverse_with_div > diverse_no_div:
                    comparison['diversity_improvements'] += 1
        
        return comparison
    
    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================
    
    def _plot_similarity_distribution(self):
        """Plot similarity score distribution."""
        all_scores = []
        for student_matches in self.matches.values():
            all_scores.extend([match['similarity_score'] for match in student_matches])
        
        if not all_scores:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(all_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(np.mean(all_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_scores):.2f}%')
        axes[0].axvline(np.median(all_scores), color='green', linestyle='--', 
                       label=f'Median: {np.median(all_scores):.2f}%')
        axes[0].set_xlabel('Similarity Score (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Similarity Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(all_scores, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1].set_ylabel('Similarity Score (%)')
        axes[1].set_title('Similarity Score Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_match_quality_distribution(self):
        """Plot match quality distribution."""
        quality_data = self.metrics['quality']['quality_distribution']
        quality_percentages = self.metrics['quality']['quality_percentages']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        qualities = list(quality_data.keys())
        counts = list(quality_data.values())
        percentages = list(quality_percentages.values())
        
        axes[0].bar(qualities, counts, color=['green', 'blue', 'orange', 'yellow'], alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Match Quality')
        axes[0].set_ylabel('Number of Matches')
        axes[0].set_title('Match Quality Distribution (Count)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
        
        # Pie chart
        axes[1].pie(percentages, labels=qualities, autopct='%1.1f%%', 
                   colors=['green', 'blue', 'orange', 'yellow'], startangle=90)
        axes[1].set_title('Match Quality Distribution (%)')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/match_quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coverage_metrics(self):
        """Plot coverage metrics."""
        coverage = self.metrics['coverage']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Coverage bar chart
        coverage_data = {
            'With Matches': coverage['students_with_matches'],
            'Without Matches': coverage['total_students'] - coverage['students_with_matches']
        }
        
        axes[0].bar(coverage_data.keys(), coverage_data.values(), 
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Number of Students')
        axes[0].set_title(f'Match Coverage ({coverage["coverage_percentage"]:.1f}%)')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (key, value) in enumerate(coverage_data.items()):
            axes[0].text(i, value, str(value), ha='center', va='bottom')
        
        # Reciprocal matches
        reciprocal_data = {
            'Reciprocal': coverage['reciprocal_matches'],
            'Non-Reciprocal': coverage['total_match_pairs'] - coverage['reciprocal_matches']
        }
        
        axes[1].bar(reciprocal_data.keys(), reciprocal_data.values(),
                   color=['blue', 'orange'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Number of Match Pairs')
        axes[1].set_title(f'Reciprocal Match Rate ({coverage["reciprocal_match_rate"]:.1f}%)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (key, value) in enumerate(reciprocal_data.items()):
            axes[1].text(i, value, str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/coverage_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_diversity_metrics(self):
        """Plot diversity metrics."""
        if not self.metrics['diversity'].get('enabled', False):
            return
        
        diversity = self.metrics['diversity']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Diversity rate
        diversity_data = {
            'Diverse Matches': diversity['diverse_matches'],
            'Non-Diverse Matches': diversity['total_matches'] - diversity['diverse_matches']
        }
        
        axes[0].bar(diversity_data.keys(), diversity_data.values(),
                   color=['purple', 'gray'], alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Number of Matches')
        axes[0].set_title(f'Diversity Rate ({diversity["diversity_rate"]:.1f}%)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Diversity by attribute
        if diversity.get('diversity_by_attribute'):
            attr_data = diversity['diversity_by_attribute']
            attributes = list(attr_data.keys())
            counts = list(attr_data.values())
            
            axes[1].bar(attributes, counts, color='teal', alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Attribute')
            axes[1].set_ylabel('Number of Diverse Matches')
            axes[1].set_title('Diversity by Attribute')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/diversity_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_analysis(self):
        """Plot cluster analysis if clustering is used."""
        clustering = self.metrics.get('clustering', {})
        if not clustering.get('clustering_used', False):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cluster sizes
        cluster_sizes = clustering['cluster_sizes']
        cluster_ids = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        axes[0].bar(cluster_ids, sizes, color='coral', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Cluster ID')
        axes[0].set_ylabel('Number of Students')
        axes[0].set_title(f'Cluster Sizes (n_clusters={clustering["n_clusters"]})')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for cluster_id, size in zip(cluster_ids, sizes):
            axes[0].text(cluster_id, size, str(size), ha='center', va='bottom')
        
        # Quality metrics
        metrics_data = {
            'Silhouette Score': clustering['silhouette_score'],
            'Inertia': clustering['inertia'] / 1000  # Scale for visualization
        }
        
        axes[1].bar(metrics_data.keys(), list(metrics_data.values()),
                   color=['blue', 'orange'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Score (Inertia scaled by 1000)')
        axes[1].set_title('Cluster Quality Metrics')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        axes[1].text(0, metrics_data['Silhouette Score'], 
                    f'{metrics_data["Silhouette Score"]:.3f}', ha='center', va='bottom')
        axes[1].text(1, metrics_data['Inertia'], 
                    f'{clustering["inertia"]:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimal_k_combined(self):
        """Plot combined normalized view of optimal k analysis."""
        k_analysis = self.metrics.get('k_analysis', {})
        if not k_analysis or 'k_values' not in k_analysis:
            return
        
        k_values = k_analysis['k_values']
        inertias = k_analysis['inertias']
        silhouette_scores = k_analysis['silhouette_scores']
        balance_scores = k_analysis['balance_scores']
        best_k = k_analysis.get('best_k', k_values[0])
        
        # Normalize all metrics to 0-1 scale
        # For inertia: lower is better, so invert (max - x)
        valid_inertias = [x for x in inertias if x != float('inf')]
        if valid_inertias and max(valid_inertias) > min(valid_inertias):
            normalized_inertia = [
                (max(valid_inertias) - x) / (max(valid_inertias) - min(valid_inertias)) 
                if x != float('inf') else 0
                for x in inertias
            ]
        else:
            normalized_inertia = [0] * len(k_values)
        
        # For silhouette: higher is better, normalize directly
        valid_silhouettes = [s for s in silhouette_scores if s >= 0]
        if valid_silhouettes and max(valid_silhouettes) > min(valid_silhouettes):
            normalized_silhouette = [
                (s - min(valid_silhouettes)) / (max(valid_silhouettes) - min(valid_silhouettes))
                if s >= 0 else 0
                for s in silhouette_scores
            ]
        else:
            normalized_silhouette = [0] * len(k_values)
        
        # Balance scores are already 0-1, but normalize for consistency
        if balance_scores and max(balance_scores) > min(balance_scores):
            normalized_balance = [
                (b - min(balance_scores)) / (max(balance_scores) - min(balance_scores))
                for b in balance_scores
            ]
        else:
            normalized_balance = balance_scores
        
        # Create the combined plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot all three normalized metrics
        ax.plot(k_values, normalized_inertia, marker='o', linestyle='-', 
               label='Inertia (normalized, higher=better)', linewidth=2.5, markersize=8, color='blue')
        ax.plot(k_values, normalized_silhouette, marker='s', linestyle='-', 
               label='Silhouette Score (normalized)', linewidth=2.5, markersize=8, color='green')
        ax.plot(k_values, normalized_balance, marker='^', linestyle='-', 
               label='Cluster Balance (normalized)', linewidth=2.5, markersize=8, color='purple')
        
        # Highlight recommended k
        best_k_idx = k_values.index(best_k) if best_k in k_values else 0
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2.5,
                  label=f'Recommended: k={best_k}', alpha=0.8)
        
        # Add a point marker at the recommended k
        ax.plot(best_k, normalized_silhouette[best_k_idx], 'ro', markersize=15, 
               markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
        ax.set_title('Combined Normalized View: Optimal k Analysis', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.set_xticks(k_values)
        
        # Add text annotation with actual values at recommended k
        if best_k_idx < len(k_values):
            inertia_val = inertias[best_k_idx] if inertias[best_k_idx] != float('inf') else 'N/A'
            sil_val = silhouette_scores[best_k_idx] if silhouette_scores[best_k_idx] >= 0 else 'N/A'
            balance_val = balance_scores[best_k_idx]
            
            if isinstance(inertia_val, float):
                inertia_str = f'{inertia_val:.2f}'
            else:
                inertia_str = str(inertia_val)
            
            if isinstance(sil_val, float):
                sil_str = f'{sil_val:.3f}'
            else:
                sil_str = str(sil_val)
            
            textstr = f'k={best_k}\nInertia: {inertia_str}\n'
            textstr += f'Silhouette: {sil_str}\n'
            textstr += f'Balance: {balance_val:.3f}'
            
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/optimal_k_combined_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance based on matching weights."""
        weights = MATCHING_WEIGHTS
        features = list(weights.keys())
        importance = list(weights.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importance, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Weight (Importance)')
        ax.set_title('Feature Importance in Matching Algorithm')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (feature, imp) in enumerate(zip(features, importance)):
            ax.text(imp, i, f'{imp}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_match_network(self):
        """Plot a network visualization of matches (for small datasets)."""
        if len(self.df) > 20:  # Skip for large datasets
            return
        
        try:
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for student_name in self.df['name']:
                G.add_node(str(student_name).strip())
            
            # Add edges with weights
            for student_name, student_matches in self.matches.items():
                for match in student_matches:
                    match_name = match['name']
                    score = match['similarity_score']
                    G.add_edge(str(student_name).strip(), str(match_name).strip(), weight=score)
            
            if len(G.nodes()) == 0:
                return
            
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw edges
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w/50 for w in weights], ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                   node_size=1000, alpha=0.9, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            
            ax.set_title('Match Network Visualization')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/match_network.png', dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("   Note: networkx not installed, skipping network visualization")
        except Exception as e:
            print(f"   Warning: Could not create network plot: {e}")
    
    def _plot_diversity_bonus_analysis(self):
        """Generate visualization plots for diversity bonus analysis."""
        if not self.diversity_bonus_stats or not self.diversity_comparison:
            return
        
        bonus_stats = self.diversity_bonus_stats
        comparison = self.diversity_comparison
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Diversity Bonus Application Rate
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Bonus application rate
        ax1 = axes[0, 0]
        if bonus_stats['total_matches'] > 0:
            bonus_rate = (bonus_stats['matches_with_bonus'] / bonus_stats['total_matches']) * 100
            no_bonus_rate = 100 - bonus_rate
            
            ax1.pie([bonus_rate, no_bonus_rate], 
                    labels=[f'With Bonus ({bonus_rate:.1f}%)', f'No Bonus ({no_bonus_rate:.1f}%)'],
                    autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
            ax1.set_title('Diversity Bonus Application Rate', fontsize=14, fontweight='bold')
        
        # Plot 2: Bonus by attribute
        ax2 = axes[0, 1]
        if bonus_stats['bonus_by_attribute']:
            attributes = list(bonus_stats['bonus_by_attribute'].keys())
            counts = list(bonus_stats['bonus_by_attribute'].values())
            
            bars = ax2.bar(attributes, counts, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
            ax2.set_title('Diversity Bonus by Attribute', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Attribute')
            ax2.set_ylabel('Number of Matches')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Plot 3: Bonus distribution
        ax3 = axes[1, 0]
        if bonus_stats['bonus_distribution']:
            ax3.hist(bonus_stats['bonus_distribution'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
            ax3.axvline(MAX_DIVERSITY_BONUS, color='r', linestyle='--', linewidth=2, label=f'Max Bonus ({MAX_DIVERSITY_BONUS})')
            ax3.set_title('Diversity Bonus Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Bonus Points')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Bonus impact on scores
        ax4 = axes[1, 1]
        if bonus_stats['bonus_impact']:
            ax4.hist(bonus_stats['bonus_impact'], bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax4.axvline(np.mean(bonus_stats['bonus_impact']), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean Impact ({np.mean(bonus_stats["bonus_impact"]):.2f})')
            ax4.set_title('Diversity Bonus Impact on Match Scores', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Score Increase (points)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/diversity_bonus_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Match Comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 5: Match changes
        ax5 = axes[0]
        if comparison['same_matches'] + comparison['different_matches'] > 0:
            total = comparison['same_matches'] + comparison['different_matches']
            same_pct = (comparison['same_matches'] / total) * 100
            diff_pct = (comparison['different_matches'] / total) * 100
            
            ax5.pie([same_pct, diff_pct],
                    labels=[f'Same Matches ({same_pct:.1f}%)', f'Different Matches ({diff_pct:.1f}%)'],
                    autopct='%1.1f%%', startangle=90, colors=['#95a5a6', '#f39c12'])
            ax5.set_title('Match Changes: With vs Without Diversity Bonus', fontsize=14, fontweight='bold')
        
        # Plot 6: Score differences
        ax6 = axes[1]
        if comparison['score_differences']:
            ax6.hist(comparison['score_differences'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax6.axvline(0, color='r', linestyle='--', linewidth=2, label='No Change')
            ax6.axvline(np.mean(comparison['score_differences']), color='g', linestyle='--', 
                       linewidth=2, label=f'Mean ({np.mean(comparison["score_differences"]):.2f})')
            ax6.set_title('Score Differences: With vs Without Bonus', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Score Difference (points)')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/match_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Diversity bonus visualizations saved to {self.output_dir}/")
    
    # ============================================================================
    # REPORT GENERATION
    # ============================================================================
    
    def _generate_report(self):
        """Generate a comprehensive text report."""
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE STUDENT MATCHING MODEL - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic Metrics
        basic = self.metrics['basic']
        report.append("BASIC METRICS")
        report.append("-" * 60)
        report.append(f"Total Students: {basic['total_students']}")
        report.append(f"Students with Matches: {basic['students_with_matches']}")
        report.append(f"Total Matches Generated: {basic['total_matches']}")
        report.append(f"Average Matches per Student: {basic['avg_matches_per_student']:.2f}")
        report.append(f"Average Similarity Score: {basic['avg_similarity_score']:.2f}%")
        report.append(f"Median Similarity Score: {basic['median_similarity_score']:.2f}%")
        report.append(f"Std Dev of Similarity Scores: {basic['std_similarity_score']:.2f}%")
        report.append(f"Min Similarity Score: {basic['min_similarity_score']:.2f}%")
        report.append(f"Max Similarity Score: {basic['max_similarity_score']:.2f}%")
        report.append("")
        
        # Coverage Metrics
        coverage = self.metrics['coverage']
        report.append("COVERAGE METRICS")
        report.append("-" * 60)
        report.append(f"Coverage Percentage: {coverage['coverage_percentage']:.2f}%")
        report.append(f"Reciprocal Match Rate: {coverage['reciprocal_match_rate']:.2f}%")
        report.append("")
        
        # Quality Metrics
        quality = self.metrics['quality']
        report.append("QUALITY METRICS")
        report.append("-" * 60)
        report.append("Match Quality Distribution:")
        for q, count in quality['quality_distribution'].items():
            pct = quality['quality_percentages'][q]
            report.append(f"  {q.capitalize()}: {count} matches ({pct:.2f}%)")
        report.append("")
        report.append("Score Percentiles:")
        for percentile, value in quality['score_distribution'].items():
            report.append(f"  {percentile.upper()}: {value:.2f}%")
        report.append("")
        
        # Diversity Metrics
        diversity = self.metrics['diversity']
        if diversity.get('enabled', False):
            report.append("DIVERSITY METRICS")
            report.append("-" * 60)
            report.append(f"Diversity Rate: {diversity['diversity_rate']:.2f}%")
            report.append(f"Diverse Matches: {diversity['diverse_matches']}/{diversity['total_matches']}")
            if diversity.get('diversity_by_attribute'):
                report.append("Diversity by Attribute:")
                for attr, count in diversity['diversity_by_attribute'].items():
                    report.append(f"  {attr}: {count} diverse matches")
            report.append("")
        
        # Diversity Bonus Analysis
        if self.diversity_bonus_stats and self.diversity_comparison:
            report.append("DIVERSITY BONUS ANALYSIS")
            report.append("-" * 60)
            bonus_stats = self.diversity_bonus_stats
            comparison = self.diversity_comparison
            
            report.append(f"Total Matches Analyzed: {bonus_stats['total_matches']}")
            report.append(f"Matches with Diversity Bonus: {bonus_stats['matches_with_bonus']}")
            if bonus_stats['total_matches'] > 0:
                bonus_rate = (bonus_stats['matches_with_bonus'] / bonus_stats['total_matches']) * 100
                report.append(f"Bonus Application Rate: {bonus_rate:.2f}%")
            report.append("")
            
            if bonus_stats['bonus_distribution']:
                report.append("Bonus Distribution:")
                report.append(f"  Mean Bonus: {np.mean(bonus_stats['bonus_distribution']):.2f} points")
                report.append(f"  Median Bonus: {np.median(bonus_stats['bonus_distribution']):.2f} points")
                report.append(f"  Max Bonus: {np.max(bonus_stats['bonus_distribution']):.2f} points")
                report.append("")
            
            if bonus_stats['bonus_impact']:
                report.append("Bonus Impact on Scores:")
                report.append(f"  Mean Score Increase: {np.mean(bonus_stats['bonus_impact']):.2f} points")
                report.append(f"  Median Score Increase: {np.median(bonus_stats['bonus_impact']):.2f} points")
                report.append("")
            
            total_comparisons = comparison['same_matches'] + comparison['different_matches']
            if total_comparisons > 0:
                report.append("Match Comparison:")
                report.append(f"  Same Matches: {comparison['same_matches']} ({comparison['same_matches']/total_comparisons*100:.1f}%)")
                report.append(f"  Different Matches: {comparison['different_matches']} ({comparison['different_matches']/total_comparisons*100:.1f}%)")
                report.append(f"  Diversity Improvements: {comparison['diversity_improvements']}")
                report.append("")
        
        # Clustering Metrics
        clustering = self.metrics.get('clustering', {})
        if clustering.get('clustering_used', False):
            report.append("CLUSTERING METRICS")
            report.append("-" * 60)
            report.append(f"Algorithm: {clustering['algorithm']}")
            if hasattr(self, 'optimal_k_used') and self.optimal_k_used:
                report.append(f"Number of Clusters: {clustering['n_clusters']} (optimal k={self.optimal_k_used} used for evaluation)")
            else:
                report.append(f"Number of Clusters: {clustering['n_clusters']}")
            report.append(f"Silhouette Score: {clustering['silhouette_score']:.3f}")
            report.append(f"Inertia: {clustering['inertia']:.2f}")
            report.append(f"Average Cluster Size: {clustering['avg_cluster_size']:.2f}")
            report.append(f"Std Dev of Cluster Sizes: {clustering['std_cluster_size']:.2f}")
            report.append("")
        
        # Optimal k Analysis
        k_analysis = self.metrics.get('k_analysis', {})
        if k_analysis and 'k_values' in k_analysis:
            report.append("OPTIMAL K ANALYSIS")
            report.append("-" * 60)
            report.append(f"Recommended k (by silhouette): {k_analysis['best_k']}")
            report.append("")
        
        # Stability Metrics
        stability = self.metrics.get('stability', {})
        if stability:
            report.append("STABILITY METRICS")
            report.append("-" * 60)
            report.append(f"Average Jaccard Similarity (across seeds): {stability['avg_jaccard_similarity']:.3f}")
            report.append(f"Number of Stability Tests: {stability['n_tests']}")
            report.append("")
        
        # Configuration
        report.append("CONFIGURATION")
        report.append("-" * 60)
        report.append(f"Matching Algorithm: {MATCHING_ALGORITHM}")
        report.append(f"Top Matches Count: {TOP_MATCHES_COUNT}")
        report.append(f"Diversity Enforcement: {ENABLE_DIVERSITY_ENFORCEMENT}")
        report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        with open(f'{self.output_dir}/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Print to console
        print("\n" + report_text)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run comprehensive evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation of Student Matching Model')
    parser.add_argument('--source', choices=['api', 'sheets', 'csv'], default='api',
                       help='Data source: api (Node.js API), sheets (Google Sheets), or csv (local file)')
    parser.add_argument('--api-url', default='http://localhost:3001/api/students',
                       help='Node.js API URL (default: http://localhost:3001/api/students)')
    parser.add_argument('--spreadsheet-id', help='Google Sheets spreadsheet ID')
    parser.add_argument('--sheet-name', default='Students', help='Google Sheets sheet name')
    parser.add_argument('--credentials', help='Path to Google service account credentials JSON')
    parser.add_argument('--csv-path', default='app_backend/data/students.csv',
                       help='Path to local CSV file')
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--skip-diversity-analysis', action='store_true',
                       help='Skip detailed diversity bonus analysis (faster)')
    
    args = parser.parse_args()
    
    # Load data based on source
    df = None
    if args.source == 'api':
        df = load_data_from_api(args.api_url)
    elif args.source == 'sheets':
        df = load_data_from_google_sheets(
            spreadsheet_id=args.spreadsheet_id,
            sheet_name=args.sheet_name,
            credentials_path=args.credentials
        )
    elif args.source == 'csv':
        df = load_data_from_csv(args.csv_path)
    
    if df is None or len(df) == 0:
        print("[ERROR] Error: Could not load student data")
        print("\nTroubleshooting:")
        print("  - For API: Make sure Node.js server is running (npm start)")
        print("  - For Sheets: Provide --spreadsheet-id and --credentials")
        print("  - For CSV: Make sure file exists at specified path")
        return
    
    print(f"\n[OK] Loaded {len(df)} students")
    print(f"   Columns: {list(df.columns)}")
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveModelEvaluator(df, output_dir=args.output_dir)
    evaluator.evaluate(include_diversity_analysis=not args.skip_diversity_analysis)


if __name__ == '__main__':
    main()

