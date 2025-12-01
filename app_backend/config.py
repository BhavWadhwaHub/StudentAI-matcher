"""
Configuration file for Student Matcher Model Accuracy and Parameters

This file contains all configurable parameters for the matching algorithm,
including weights, thresholds, and accuracy settings.
"""

# ============================================================================
# MATCHING WEIGHTS
# ============================================================================
# These weights determine how much each attribute contributes to the similarity score
# Optimized for K-means clustering with balanced emphasis on key factors
# Total sums to 100 for accurate percentage calculation
# Weights are designed to:
# 1. Prioritize academic compatibility (major)
# 2. Consider interpersonal fit (personality)
# 3. Reward shared interests (activities, hobbies)
# 4. Account for cultural and communication factors (reduced to promote diversity)
# NOTE: Country and language weights reduced to minimize cultural clustering
MATCHING_WEIGHTS = {
    'major': 30,          # Academic major (highest weight - critical for study compatibility)
    'personality': 25,    # Personality type (important for interpersonal compatibility)
    'interests': 22,      # Hobbies and interests (strong indicator of shared activities) - Jaccard Index
    'country': 5,         # Country of origin (reduced from 10% to promote diversity)
    'year': 6,            # Academic year (similar experience level and goals)
    'studyStyle': 7,      # Study style preference (learning compatibility) - increased from 5%
    'language': 1,        # Language spoken (reduced from 2% to minimize clustering)
    'cuisine': 2,         # Food preferences (social compatibility indicator) - increased from 1%
    'movies': 2           # Movie preferences (entertainment compatibility) - increased from 1%
    # Total: 100
}

# ============================================================================
# SIMILARITY THRESHOLDS
# ============================================================================
# These thresholds determine match quality labels (0-100 scale)
SIMILARITY_THRESHOLDS = {
    'excellent': 80,      # 80-100%: Excellent match
    'very_good': 60,      # 60-79%: Very good match
    'good': 40,           # 40-59%: Good match
    'decent': 0           # 0-39%: Decent match (minimum threshold)
}

# ============================================================================
# MATCHING PARAMETERS
# ============================================================================
# Number of top matches to return per student
TOP_MATCHES_COUNT = 3

# Minimum similarity score to include in results (0-100)
# Dynamic threshold based on number of students:
# - With 2-5 students: 25% (low threshold to ensure matches)
# - With 6-10 students: 35% (moderate threshold)
# - With 11+ students: 38% (balanced - lowered to help unique students find matches)
# This ensures matches are found even with small datasets and unique profiles
MIN_SIMILARITY_SCORE = 25  # Base minimum, will be adjusted dynamically

# Enable/disable random shuffling for tie-breaking
ENABLE_TIE_BREAKING = True

# Random seed for reproducibility (set to None for non-deterministic)
RANDOM_SEED = 42

# ============================================================================
# ACCURACY SETTINGS
# ============================================================================
# Decimal precision for similarity scores
SIMILARITY_SCORE_PRECISION = 2

# Decimal precision for final match scores
FINAL_SCORE_PRECISION = 1

# Enable detailed logging for debugging
ENABLE_DEBUG_LOGGING = True

# ============================================================================
# CLUSTERING SETTINGS
# ============================================================================
# Matching algorithm to use: 'weighted_similarity', 'kmeans', or 'hybrid'
# - 'weighted_similarity': Pairwise similarity matching (current)
# - 'kmeans': K-means clustering with weighted features
# - 'hybrid': K-means clustering + weighted similarity within clusters
MATCHING_ALGORITHM = 'kmeans'  # Default to K-means clustering

# Number of clusters for K-means (None = auto-determine using elbow method)
KMEANS_N_CLUSTERS = None

# Minimum number of clusters (for auto-determination)
KMEANS_MIN_CLUSTERS = 2

# Maximum number of clusters (for auto-determination)
KMEANS_MAX_CLUSTERS = 20

# Maximum matches to return from same cluster
MAX_MATCHES_FROM_SAME_CLUSTER = 2

# Enable cross-cluster matching (match students from different clusters)
ENABLE_CROSS_CLUSTER_MATCHING = True

# ============================================================================
# JACCARD INDEX SETTINGS
# ============================================================================
# Minimum items required in a list for Jaccard Index calculation
MIN_ITEMS_FOR_JACCARD = 1

# ============================================================================
# DIVERSITY SETTINGS
# ============================================================================
# Enable diversity enforcement in matching
ENABLE_DIVERSITY_ENFORCEMENT = True

# Diversity bonus points (added to similarity score for diverse matches)
# Applied when students differ in key attributes
DIVERSITY_BONUS = {
    'country': 3.0,       # Bonus for different countries (promotes cultural diversity)
    'major': 2.0,         # Bonus for different majors (promotes academic diversity)
    'language': 1.0,      # Bonus for different languages (promotes linguistic diversity)
    'year': 0.5           # Small bonus for different years (promotes experience diversity)
}

# Maximum diversity bonus that can be applied (cap to prevent over-weighting)
MAX_DIVERSITY_BONUS = 5.0

# Minimum number of diverse matches required per student
# A diverse match is one that differs in at least one key attribute
MIN_DIVERSE_MATCHES = 1  # At least 1 of TOP_MATCHES_COUNT should be diverse

# Attributes to consider for diversity enforcement
DIVERSITY_ATTRIBUTES = ['country', 'major', 'language']

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================
# Required columns for matching
REQUIRED_COLUMNS = [
    'name', 'major', 'year', 'language', 'country', 
    'personality', 'studyStyle', 'cuisine', 'interests', 'movies'
]

# Multi-value fields that use Jaccard Index
MULTI_VALUE_FIELDS = ['cuisine', 'interests', 'movies']

# Categorical fields that use exact matching
CATEGORICAL_FIELDS = ['major', 'country', 'personality', 'year', 'language', 'studyStyle']

# ============================================================================
# FALLBACK SETTINGS
# ============================================================================
# Fallback weights (used if main matching fails)
# Updated to match reduced cultural clustering in main weights
FALLBACK_WEIGHTS = {
    'major': 20,
    'year': 10,
    'language': 5,        # Reduced from 15
    'country': 5,         # Reduced from 10
    'personality': 20,    # Increased from 15
    'studyStyle': 10,
    'cuisine': 10,
    'interests': 18,      # Increased from 15
    'movies': 2           # Reduced from 5
}

