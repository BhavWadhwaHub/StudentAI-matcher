import pandas as pd
import random
import ast
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from config import (
    MATCHING_WEIGHTS, SIMILARITY_THRESHOLDS, TOP_MATCHES_COUNT,
    MIN_SIMILARITY_SCORE, ENABLE_TIE_BREAKING, RANDOM_SEED,
    SIMILARITY_SCORE_PRECISION, FINAL_SCORE_PRECISION,
    REQUIRED_COLUMNS, MULTI_VALUE_FIELDS, CATEGORICAL_FIELDS,
    FALLBACK_WEIGHTS, ENABLE_DEBUG_LOGGING,
    MATCHING_ALGORITHM, KMEANS_N_CLUSTERS, KMEANS_MIN_CLUSTERS,
    KMEANS_MAX_CLUSTERS, MAX_MATCHES_FROM_SAME_CLUSTER,
    ENABLE_CROSS_CLUSTER_MATCHING,
    ENABLE_DIVERSITY_ENFORCEMENT, DIVERSITY_BONUS, MAX_DIVERSITY_BONUS,
    MIN_DIVERSE_MATCHES, DIVERSITY_ATTRIBUTES
)

def jaccard_index(set1, set2):
    """
    Calculate Jaccard Index between two sets.
    
    Args:
        set1, set2: Sets to compare
    
    Returns:
        float: Jaccard Index (0-1)
    """
    if not set1 and not set2:
        return 1.0  # Both empty sets are considered identical
    if not set1 or not set2:
        return 0.0  # One empty, one not
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_diversity_bonus(student1, student2):
    """
    Calculate diversity bonus for matches that differ in key attributes.
    Promotes diversity by rewarding matches across different countries, majors, languages, etc.
    
    Args:
        student1, student2: Student data dictionaries
    
    Returns:
        float: Diversity bonus (0 to MAX_DIVERSITY_BONUS)
    """
    if not ENABLE_DIVERSITY_ENFORCEMENT:
        return 0.0
    
    bonus = 0.0
    
    # Check each diversity attribute
    for attr in DIVERSITY_ATTRIBUTES:
        if attr not in DIVERSITY_BONUS:
            continue
        
        # Safely get attribute values
        if hasattr(student1, 'get'):
            val1 = student1.get(attr)
        else:
            val1 = student1[attr] if attr in student1 else None
        
        if hasattr(student2, 'get'):
            val2 = student2.get(attr)
        else:
            val2 = student2[attr] if attr in student2 else None
        
        # Skip if either value is None/NaN
        if val1 is None or val2 is None:
            continue
        try:
            if pd.isna(val1) or pd.isna(val2):
                continue
        except (TypeError, ValueError):
            pass
        
        # Check if values are different (promotes diversity)
        if val1 != val2:
            bonus += DIVERSITY_BONUS[attr]
    
    # Cap the bonus at maximum
    return min(bonus, MAX_DIVERSITY_BONUS)


def is_diverse_match(student1, student2):
    """
    Check if two students form a diverse match (differ in at least one key attribute).
    
    Args:
        student1, student2: Student data dictionaries
    
    Returns:
        bool: True if match is diverse, False otherwise
    """
    if not ENABLE_DIVERSITY_ENFORCEMENT:
        return False
    
    for attr in DIVERSITY_ATTRIBUTES:
        # Safely get attribute values
        if hasattr(student1, 'get'):
            val1 = student1.get(attr)
        else:
            val1 = student1[attr] if attr in student1 else None
        
        if hasattr(student2, 'get'):
            val2 = student2.get(attr)
        else:
            val2 = student2[attr] if attr in student2 else None
        
        # Skip if either value is None/NaN
        if val1 is None or val2 is None:
            continue
        try:
            if pd.isna(val1) or pd.isna(val2):
                continue
        except (TypeError, ValueError):
            pass
        
        # If values differ, this is a diverse match
        if val1 != val2:
            return True
    
    return False


def enforce_diversity_in_matches(student_matches, student_data, df_or_dict):
    """
    Enforce diversity in match results by ensuring minimum number of diverse matches.
    If not enough diverse matches exist, promotes diverse matches in the list.
    
    Args:
        student_matches: List of tuples (match_name, similarity_score, is_same_cluster)
        student_data: Dictionary with student data
        df_or_dict: DataFrame with all student data OR dictionary mapping names to student data
    
    Returns:
        List of tuples with diversity-enforced matches
    """
    if not ENABLE_DIVERSITY_ENFORCEMENT or not student_matches:
        return student_matches
    
    # Handle both DataFrame and dictionary inputs for backward compatibility
    if isinstance(df_or_dict, dict):
        student_data_dict = df_or_dict
        get_match_data = lambda name: student_data_dict.get(name)
    else:
        df = df_or_dict
        get_match_data = lambda name: (df[df['name'].astype(str).str.strip() == name].iloc[0].to_dict() 
                                       if len(df[df['name'].astype(str).str.strip() == name]) > 0 else None)
    
    # Separate matches into diverse and non-diverse
    diverse_matches = []
    non_diverse_matches = []
    
    for match in student_matches:
        match_name = match[0]
        # Get match data using optimized lookup
        match_data = get_match_data(match_name)
        if match_data is None:
            non_diverse_matches.append(match)
            continue
        
        if is_diverse_match(student_data, match_data):
            diverse_matches.append(match)
        else:
            non_diverse_matches.append(match)
    
    # Ensure minimum diverse matches
    required_diverse = min(MIN_DIVERSE_MATCHES, TOP_MATCHES_COUNT)
    
    if len(diverse_matches) < required_diverse and len(non_diverse_matches) > 0:
        # Need to promote some non-diverse matches or find more diverse ones
        # For now, we'll prioritize diverse matches in the final list
        # In a more sophisticated implementation, we could search for additional diverse matches
        if ENABLE_DEBUG_LOGGING:
            print(f"⚠️  Warning: Only {len(diverse_matches)} diverse matches found, need at least {required_diverse}")
    
    # Sort both lists by score (descending) to maintain quality
    diverse_matches.sort(key=lambda x: x[1], reverse=True)
    non_diverse_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Combine: diverse matches first (sorted by score), then non-diverse (sorted by score)
    # This ensures diverse matches are prioritized while maintaining score-based quality
    final_matches = diverse_matches + non_diverse_matches
    
    # Ensure we return exactly TOP_MATCHES_COUNT matches
    return final_matches[:TOP_MATCHES_COUNT]


def calculate_weighted_similarity(student1, student2, include_diversity_bonus=True):
    """
    Calculate weighted similarity score between two students using specified weights.
    Can optionally include diversity bonus for matches that differ in key attributes.
    
    Args:
        student1, student2: Student data dictionaries
        include_diversity_bonus: If True, adds diversity bonus to the score
    
    Returns:
        float: Weighted similarity score (0-100, can exceed 100 with diversity bonus if enabled)
    """
    # Use weights from configuration
    weights = MATCHING_WEIGHTS
    
    total_score = 0.0
    total_weight = sum(weights.values())
    
    # Single-value categorical fields (exact match)
    for field in CATEGORICAL_FIELDS:
        if field in student1 and field in student2 and field in weights:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Check for exact match and non-empty values
            if val1 == val2 and val1:
                total_score += weights[field]
    
    # Multi-value fields using Jaccard Index
    for field in MULTI_VALUE_FIELDS:
        if field in student1 and field in student2 and field in weights:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Convert to sets safely
            if isinstance(val1, list):
                list1 = set(val1)
            elif isinstance(val1, str) and val1.strip():
                # Handle string representations of lists
                try:
                    list1 = set(ast.literal_eval(val1))
                except (ValueError, SyntaxError):
                    list1 = set()
            else:
                list1 = set()
            
            if isinstance(val2, list):
                list2 = set(val2)
            elif isinstance(val2, str) and val2.strip():
                # Handle string representations of lists
                try:
                    list2 = set(ast.literal_eval(val2))
                except (ValueError, SyntaxError):
                    list2 = set()
            else:
                list2 = set()
            
            jaccard = jaccard_index(list1, list2)
            total_score += jaccard * weights[field]
    
    # Convert to percentage (0-100)
    similarity_percentage = (total_score / total_weight) * 100 if total_weight > 0 else 0.0
    
    # Add diversity bonus if enabled (applied after normalization in matching functions)
    # But include it here for threshold checking to help diverse matches pass the threshold
    if include_diversity_bonus and ENABLE_DIVERSITY_ENFORCEMENT:
        diversity_bonus = calculate_diversity_bonus(student1, student2)
        similarity_percentage += diversity_bonus
    
    return round(similarity_percentage, SIMILARITY_SCORE_PRECISION)


def generate_matches(df):
    """
    Generate student matches using weighted similarity scoring with Jaccard Index.
    
    Args:
        df: DataFrame with student data including name, email, major, year, language, 
            country, personality, studyStyle, cuisine, interests, movies
    
    Returns:
        dict: Dictionary mapping student names to their top 3 matches with detailed info
    """
    # Input validation
    if df is None or len(df) == 0:
        return {}
    
    if len(df) == 1:
        return {}
    
    # Defensive copy
    df = df.copy()
    
    # Normalize column names to handle different naming conventions
    column_mapping = {
        'Name': 'name',
        'Email': 'email', 
        'Major': 'major',
        'Year': 'year',
        'Language': 'language',
        'Country': 'country',
        'Personality': 'personality',
        'Study Style': 'studyStyle',
        'Cuisine': 'cuisine',
        'Interests': 'interests',
        'Movies': 'movies'
    }
    
    # Rename columns to standardize
    df = df.rename(columns=column_mapping)
    
    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if ENABLE_DEBUG_LOGGING:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
        return {}

    try:
        # Ensure multi-value fields are lists and handle string representations
        for field in MULTI_VALUE_FIELDS:
            if field in df.columns:
                def safe_eval(x):
                    if isinstance(x, list):
                        return x
                    elif isinstance(x, str) and x.strip():
                        try:
                            # Use ast.literal_eval for safer evaluation
                            return ast.literal_eval(x)
                        except (ValueError, SyntaxError):
                            # If parsing fails, treat as single item
                            return [x.strip()]
                    else:
                        return []
                
                df[field] = df[field].apply(safe_eval)

        # Choose matching algorithm based on configuration
        if MATCHING_ALGORITHM == 'kmeans':
            if ENABLE_DEBUG_LOGGING:
                print("Using K-means clustering with weighted features")
            return _kmeans_clustering_matching(df)
        elif MATCHING_ALGORITHM == 'hybrid':
            if ENABLE_DEBUG_LOGGING:
                print("Using hybrid approach: K-means clustering + weighted similarity")
            return _hybrid_clustering_matching(df)
        else:
            # Default to weighted similarity matching
            if ENABLE_DEBUG_LOGGING:
                print("Using weighted similarity scoring with Jaccard Index")
            return _weighted_similarity_matching(df)

    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error in generate_matches: {e}")
        return _fallback_similarity_matching(df)


def _weighted_similarity_matching(df):
    """
    Main matching function using weighted similarity scoring with Jaccard Index.
    Optimized for performance with pre-computed student data dictionary.
    """
    try:
        # Set random seed for reproducibility if specified
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        
        # Pre-compute student data dictionary once (major performance optimization)
        # This avoids repeated DataFrame filtering and dict conversions
        student_data_dict = {}
        for idx, row in df.iterrows():
            try:
                name = row.get('name') if hasattr(row, 'get') else row['name']
                if name is None:
                    continue
                try:
                    if pd.isna(name):
                        continue
                except (TypeError, ValueError):
                    pass
                name = str(name).strip()
                if name:
                    student_data_dict[name] = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            except (KeyError, TypeError, AttributeError):
                continue
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Pre-computed data for {len(student_data_dict)} students")
        
        matches = {}
        student_names_list = list(student_data_dict.keys())
        dynamic_threshold = _get_dynamic_threshold(len(df))
        
        # Use indexed iteration instead of nested iterrows() for better performance
        for i, student1_name in enumerate(student_names_list):
            student1_data = student_data_dict[student1_name]
            student_matches = []
            
            # Compare with all other students (maintains same logic as before)
            for j, student2_name in enumerate(student_names_list):
                if i != j:  # Don't match with self
                    student2_data = student_data_dict[student2_name]
                    
                    score = calculate_weighted_similarity(student1_data, student2_data)
                    
                    if score >= dynamic_threshold:
                        student_matches.append((student2_name, score))
            
            # Normalize scores to 0-100 range based on actual distribution
            # Convert to format expected by _normalize_scores (add is_same_cluster=False)
            if student_matches:
                student_matches_with_cluster = [(name, score, False) for name, score in student_matches]
                normalized_matches = _normalize_scores(student_matches_with_cluster, student1_data, student_data_dict)
                student_matches = [(name, score) for name, score, _ in normalized_matches]
            
            # Sort by normalized score and take top N matches
            student_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Enforce diversity in matches
            if ENABLE_DIVERSITY_ENFORCEMENT:
                student_matches_with_cluster = [(name, score, False) for name, score in student_matches]
                student_matches_with_cluster = enforce_diversity_in_matches(
                    student_matches_with_cluster, student1_data, student_data_dict
                )
                student_matches = [(name, score) for name, score, _ in student_matches_with_cluster]
            
            top_matches = student_matches[:TOP_MATCHES_COUNT]
            
            # Fair tie-breaking if enabled
            if ENABLE_TIE_BREAKING and len(top_matches) > 1:
                random.shuffle(top_matches)
            
            # Convert to detailed match format
            detailed_matches = []
            for match_name, similarity_score in top_matches:
                try:
                    match_info = _create_detailed_match(student_data_dict, student1_name, match_name, similarity_score)
                    detailed_matches.append(match_info)
                except Exception as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Error creating detailed match for {student1_name} -> {match_name}: {e}")
                    continue
            
            matches[student1_name] = detailed_matches
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Generated weighted similarity matches for {len(matches)} students")
        return matches
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error in weighted similarity matching: {e}")
        return {}


def _get_dynamic_threshold(num_students):
    """
    Calculate dynamic similarity threshold based on number of students.
    
    Args:
        num_students: Number of students in the database
    
    Returns:
        float: Dynamic threshold (0-100)
    """
    if num_students <= 5:
        return 25.0  # Low threshold for small datasets (2-5 students)
    elif num_students <= 10:
        return 35.0  # Moderate threshold (6-10 students)
    else:
        return 38.0  # Balanced threshold (11+ students) - lowered to help unique students


def _normalize_scores(student_matches, student_data=None, df_or_dict=None):
    """
    Normalize similarity scores to 0-100 range based on actual score distribution.
    Maps the actual min-max range to 0-100 so the best match is 100% and worst is 0%.
    Applies diversity bonus AFTER normalization to preserve its impact.
    
    Args:
        student_matches: List of tuples (match_name, similarity_score, is_same_cluster)
        student_data: Optional student data dict for diversity bonus calculation
        df_or_dict: Optional DataFrame OR dictionary for looking up match data
    
    Returns:
        List of tuples with normalized scores (match_name, normalized_score, is_same_cluster)
    """
    if not student_matches or len(student_matches) == 0:
        return []
    
    # Handle both DataFrame and dictionary inputs for backward compatibility
    if isinstance(df_or_dict, dict):
        student_data_dict = df_or_dict
        get_match_data = lambda name: student_data_dict.get(name)
    elif df_or_dict is not None:
        df = df_or_dict
        get_match_data = lambda name: (df[df['name'].astype(str).str.strip() == name].iloc[0].to_dict() 
                                       if len(df[df['name'].astype(str).str.strip() == name]) > 0 else None)
    else:
        get_match_data = lambda name: None
    
    # Extract scores (these may include diversity bonuses from threshold checking)
    scores = [match[1] for match in student_matches]
    
    if len(scores) == 1:
        # Single match: set to 100%, then apply diversity bonus if enabled
        normalized_score = 100.0
        if ENABLE_DIVERSITY_ENFORCEMENT and student_data is not None and df_or_dict is not None:
            match_name = student_matches[0][0]
            match_data = get_match_data(match_name)
            if match_data is not None:
                diversity_bonus = calculate_diversity_bonus(student_data, match_data)
                # Apply diversity bonus as percentage boost (scaled to 0-100 range)
                normalized_score = min(100.0, normalized_score + (diversity_bonus * 0.5))  # Scale bonus
        return [(student_matches[0][0], round(normalized_score, SIMILARITY_SCORE_PRECISION), student_matches[0][2])]
    
    # Find min and max scores
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        # All scores are the same: set all to 100%, then apply diversity bonuses
        normalized_matches = []
        for match_name, _, is_same_cluster in student_matches:
            normalized_score = 100.0
            if ENABLE_DIVERSITY_ENFORCEMENT and student_data is not None and df_or_dict is not None:
                match_data = get_match_data(match_name)
                if match_data is not None:
                    diversity_bonus = calculate_diversity_bonus(student_data, match_data)
                    # Apply diversity bonus as percentage boost
                    normalized_score = min(100.0, normalized_score + (diversity_bonus * 0.5))
            normalized_matches.append((match_name, round(normalized_score, SIMILARITY_SCORE_PRECISION), is_same_cluster))
        return normalized_matches
    
    # Normalize scores: map [min_score, max_score] to [0, 100]
    # Formula: normalized = ((score - min_score) / (max_score - min_score)) * 100
    normalized_matches = []
    for match_name, score, is_same_cluster in student_matches:
        normalized_score = ((score - min_score) / score_range) * 100.0
        
        # Apply diversity bonus AFTER normalization to preserve its impact
        # This ensures diverse matches get a visible boost in the final ranking
        if ENABLE_DIVERSITY_ENFORCEMENT and student_data is not None and df_or_dict is not None:
            match_data = get_match_data(match_name)
            if match_data is not None:
                diversity_bonus = calculate_diversity_bonus(student_data, match_data)
                # Scale diversity bonus to 0-100 range (multiply by 0.5 to make it impactful but not overwhelming)
                # This gives up to 2.5% boost (5.0 max bonus * 0.5 = 2.5%)
                diversity_boost = diversity_bonus * 0.5
                normalized_score = min(100.0, normalized_score + diversity_boost)
        
        normalized_score = round(normalized_score, SIMILARITY_SCORE_PRECISION)
        normalized_matches.append((match_name, normalized_score, is_same_cluster))
    
    if ENABLE_DEBUG_LOGGING and len(student_matches) > 0:
        print(f"   Score normalization: min={min_score:.2f}%, max={max_score:.2f}% -> normalized to 0-100%")
        if ENABLE_DIVERSITY_ENFORCEMENT:
            print(f"   Diversity bonuses applied after normalization")
    
    return normalized_matches


def _safe_is_none_or_nan(val):
    """
    Safely check if a value is None or NaN, handling arrays and lists.
    
    Args:
        val: Value to check
    
    Returns:
        bool: True if value is None or NaN, False otherwise
    """
    if val is None:
        return True
    try:
        # Check if it's a scalar value that can be checked with pd.isna
        if isinstance(val, (list, np.ndarray)):
            # For arrays/lists, check if all elements are None/NaN
            if len(val) == 0:
                return True
            # Don't use pd.isna on arrays directly
            return False
        # For scalar values, use pd.isna
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def encode_multilabel(df, column):
    """
    Multi-label encode a column using MultiLabelBinarizer.
    Handles string representations of lists and ensures all values are lists.
    
    Args:
        df: DataFrame with the column to encode
        column: Column name to encode
    
    Returns:
        tuple: (encoded_matrix, binarizer) where encoded_matrix is a numpy array
    """
    try:
        # Convert column values to lists
        list_values = []
        for val in df[column]:
            if isinstance(val, list):
                list_values.append([str(v).strip() for v in val if v])
            elif isinstance(val, str) and val.strip():
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        list_values.append([str(v).strip() for v in parsed if v])
                    else:
                        list_values.append([str(parsed).strip()])
                except (ValueError, SyntaxError):
                    # Treat as single item
                    list_values.append([val.strip()])
            else:
                list_values.append([])
        
        # Fit and transform using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(list_values)
        
        return encoded, mlb
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error encoding multilabel column {column}: {e}")
        return np.array([]), None


def preprocess_data(df):
    """
    Preprocess student data for clustering:
    - Drop Name and Email before clustering (they remain in original df)
    - One-hot encode: Major, Year, Country, Personality, Study Style, Cuisine
    - Multi-label encode: Language, Interests, Movies
    - Standardize numeric fields if any exist
    
    Args:
        df: DataFrame with student data
    
    Returns:
        tuple: (preprocessed_df, preprocessing_info) where preprocessing_info contains
               encoders, scalers, and metadata needed for inverse transformation
    """
    try:
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Store original Name and Email columns (they're not used in clustering)
        if 'name' in df_copy.columns:
            name_col = df_copy['name'].copy()
        elif 'Name' in df_copy.columns:
            name_col = df_copy['Name'].copy()
            df_copy = df_copy.rename(columns={'Name': 'name'})
        else:
            name_col = None
        
        if 'email' in df_copy.columns:
            email_col = df_copy['email'].copy()
        elif 'Email' in df_copy.columns:
            email_col = df_copy['Email'].copy()
            df_copy = df_copy.rename(columns={'Email': 'email'})
        else:
            email_col = None
        
        # Normalize column names
        column_mapping = {
            'Name': 'name',
            'Email': 'email',
            'Major': 'major',
            'Year': 'year',
            'Language': 'language',
            'Country': 'country',
            'Personality': 'personality',
            'Study Style': 'studyStyle',
            'Cuisine': 'cuisine',
            'Interests': 'interests',
            'Movies': 'movies'
        }
        df_copy = df_copy.rename(columns=column_mapping)
        
        # Initialize preprocessing info dictionary
        preprocessing_info = {
            'name_col': name_col,
            'email_col': email_col,
            'onehot_encoders': {},
            'multilabel_encoders': {},
            'scaler': None,
            'numeric_columns': []
        }
        
        # Prepare dataframe for clustering (drop Name and Email)
        clustering_df = df_copy.drop(columns=['name', 'email'], errors='ignore')
        
        # One-hot encode single-choice categorical fields
        onehot_fields = ['major', 'year', 'country', 'personality', 'studyStyle', 'cuisine']
        onehot_encoded_dfs = []
        
        for field in onehot_fields:
            if field in clustering_df.columns:
                # Convert list values to strings for one-hot encoding
                def convert_to_string(val):
                    if isinstance(val, list):
                        # For lists, take first value or join if multiple
                        return str(val[0]).strip() if val else ''
                    elif isinstance(val, str) and val.strip():
                        # Check if it's a string representation of a list
                        try:
                            parsed = ast.literal_eval(val)
                            if isinstance(parsed, list):
                                return str(parsed[0]).strip() if parsed else ''
                            return str(parsed).strip()
                        except (ValueError, SyntaxError):
                            return str(val).strip()
                    elif val is not None:
                        return str(val).strip()
                    return ''
                
                # Convert column to strings
                clustering_df[field] = clustering_df[field].apply(convert_to_string)
                
                # One-hot encode
                onehot = pd.get_dummies(clustering_df[field], prefix=field, dummy_na=False)
                onehot_encoded_dfs.append(onehot)
                preprocessing_info['onehot_encoders'][field] = list(onehot.columns)
        
        # Multi-label encode multi-select fields
        multilabel_fields = ['language', 'interests', 'movies']
        multilabel_encoded_dfs = []
        
        for field in multilabel_fields:
            if field in clustering_df.columns:
                encoded, mlb = encode_multilabel(clustering_df, field)
                if encoded.size > 0 and mlb is not None:
                    # Create DataFrame with column names
                    mlb_df = pd.DataFrame(
                        encoded,
                        columns=[f"{field}_{val}" for val in mlb.classes_],
                        index=clustering_df.index
                    )
                    multilabel_encoded_dfs.append(mlb_df)
                    preprocessing_info['multilabel_encoders'][field] = mlb
        
        # Drop original columns that were encoded
        columns_to_drop = onehot_fields + multilabel_fields
        clustering_df = clustering_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Combine all encoded DataFrames
        all_encoded_dfs = onehot_encoded_dfs + multilabel_encoded_dfs
        if all_encoded_dfs:
            encoded_df = pd.concat(all_encoded_dfs, axis=1)
        else:
            encoded_df = pd.DataFrame()
        
        # Add any remaining numeric columns
        numeric_cols = clustering_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            preprocessing_info['numeric_columns'] = numeric_cols
            numeric_df = clustering_df[numeric_cols]
            
            # Standardize numeric fields
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(numeric_df)
            scaled_numeric_df = pd.DataFrame(
                scaled_numeric,
                columns=numeric_cols,
                index=clustering_df.index
            )
            preprocessing_info['scaler'] = scaler
            
            # Combine with encoded features
            if encoded_df.size > 0:
                preprocessed_df = pd.concat([encoded_df, scaled_numeric_df], axis=1)
            else:
                preprocessed_df = scaled_numeric_df
        else:
            # No numeric columns, use only encoded features
            preprocessed_df = encoded_df
        
        # Fill any NaN values with 0
        preprocessed_df = preprocessed_df.fillna(0)
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Preprocessed data shape: {preprocessed_df.shape}")
            print(f"Features: {len(preprocessed_df.columns)}")
        
        return preprocessed_df, preprocessing_info
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
        return None, None


def find_best_k(df, min_k=2, max_k=12):
    """
    Determine optimal number of clusters using the Elbow method.
    Tests K from min_k to max_k, computes inertia, and chooses elbow point.
    
    Args:
        df: Preprocessed DataFrame (feature matrix)
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test
    
    Returns:
        int: Optimal number of clusters (elbow point)
    """
    try:
        if len(df) < min_k:
            return min(len(df), min_k)
        
        max_k = min(max_k, len(df) - 1)
        if max_k < min_k:
            return min_k
        
        # Convert DataFrame to numpy array if needed
        if isinstance(df, pd.DataFrame):
            feature_matrix = df.values
        else:
            feature_matrix = df
        
        # Calculate inertia for each k
        inertias = []
        k_values = range(min_k, max_k + 1)
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                kmeans.fit(feature_matrix)
                inertias.append(kmeans.inertia_)
            except Exception as e:
                if ENABLE_DEBUG_LOGGING:
                    print(f"Error computing inertia for k={k}: {e}")
                inertias.append(float('inf'))
        
        # Find elbow point using rate of change
        if len(inertias) < 2:
            return min_k
        
        # Calculate rate of change (derivative approximation)
        # Elbow is where the rate of change decreases significantly
        best_k = min_k
        max_improvement = 0
        
        for i in range(1, len(inertias)):
            # Improvement in inertia reduction
            improvement = inertias[i-1] - inertias[i]
            # Normalized by previous inertia
            if inertias[i-1] > 0:
                normalized_improvement = improvement / inertias[i-1]
                if normalized_improvement > max_improvement:
                    max_improvement = normalized_improvement
                    best_k = k_values[i-1]
        
        # If no clear elbow, use k with lowest inertia (but prefer smaller k)
        if max_improvement == 0:
            # Use silhouette score as tie-breaker
            best_silhouette = -1
            for k in k_values:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                    labels = kmeans.fit_predict(feature_matrix)
                    if len(set(labels)) > 1:
                        silhouette = silhouette_score(feature_matrix, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_k = k
                except Exception:
                    continue
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Elbow method: Optimal k={best_k} (tested {min_k}-{max_k})")
            print(f"Inertias: {dict(zip(k_values, inertias))}")
        
        return best_k
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error finding best k: {e}")
            import traceback
            traceback.print_exc()
        return min_k


def run_kmeans(df, n_clusters=None):
    """
    Run K-Means clustering on preprocessed data.
    
    Args:
        df: Preprocessed DataFrame (feature matrix)
        n_clusters: Number of clusters (if None, uses find_best_k)
    
    Returns:
        tuple: (labels, kmeans_model) where labels is array of cluster assignments
    """
    try:
        # Convert DataFrame to numpy array if needed
        if isinstance(df, pd.DataFrame):
            feature_matrix = df.values
        else:
            feature_matrix = df
        
        if len(feature_matrix) == 0:
            return None, None
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = find_best_k(feature_matrix, min_k=2, max_k=12)
        
        # Ensure we have at least 2 students for clustering (can't cluster 1 student)
        if len(feature_matrix) < 2:
            if ENABLE_DEBUG_LOGGING:
                print(f"Warning: Need at least 2 students for clustering, got {len(feature_matrix)}")
            return None, None
        
        n_clusters = min(n_clusters, len(feature_matrix))
        
        # Ensure at least 2 clusters (can't have 1 cluster for matching)
        if n_clusters < 2 and len(feature_matrix) >= 2:
            n_clusters = 2
        
        # Run K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
        if ENABLE_DEBUG_LOGGING:
            print(f"K-Means clustering: {len(feature_matrix)} students in {n_clusters} clusters")
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"  Cluster {label}: {count} students")
        
        return labels, kmeans
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error running K-Means: {e}")
            import traceback
            traceback.print_exc()
        return None, None


def prepare_output(original_df, labels, preprocessing_info):
    """
    Attach cluster labels back to original unprocessed dataframe.
    Returns dataframe with Name, Email, Cluster Number, and optionally similar cluster members.
    
    Args:
        original_df: Original DataFrame with Name and Email
        labels: Cluster labels from K-Means
        preprocessing_info: Dictionary containing preprocessing metadata
    
    Returns:
        DataFrame: Original dataframe with cluster labels added
    """
    try:
        # Make a copy of original dataframe
        output_df = original_df.copy()
        
        # Normalize column names
        column_mapping = {
            'Name': 'name',
            'Email': 'email',
            'Major': 'major',
            'Year': 'year',
            'Language': 'language',
            'Country': 'country',
            'Personality': 'personality',
            'Study Style': 'studyStyle',
            'Cuisine': 'cuisine',
            'Interests': 'interests',
            'Movies': 'movies'
        }
        output_df = output_df.rename(columns=column_mapping)
        
        # Ensure we have name column - align with dataframe index
        if 'name' not in output_df.columns and preprocessing_info.get('name_col') is not None:
            name_col = preprocessing_info['name_col']
            if isinstance(name_col, pd.Series):
                # Align with output_df index
                output_df['name'] = name_col.reindex(output_df.index).values
            else:
                output_df['name'] = name_col.values if hasattr(name_col, 'values') else name_col
        
        # Add cluster labels - align with dataframe index
        if labels is not None and len(labels) == len(output_df):
            # Convert labels to Series with matching index
            output_df['cluster'] = pd.Series(labels, index=output_df.index)
        else:
            if ENABLE_DEBUG_LOGGING:
                print(f"Warning: Label length {len(labels) if labels is not None else 0} != DataFrame length {len(output_df)}")
            output_df['cluster'] = -1
        
        # Reorder columns to have Name, Email, Cluster first
        columns = ['name', 'email', 'cluster']
        other_columns = [col for col in output_df.columns if col not in columns]
        output_df = output_df[columns + other_columns]
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Output DataFrame shape: {output_df.shape}")
            print(f"Columns: {list(output_df.columns)}")
        
        return output_df
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error preparing output: {e}")
            import traceback
            traceback.print_exc()
        return original_df


def _prepare_features_for_clustering(df):
    """
    Convert student data to numerical features for K-means clustering.
    Uses weighted encoding based on MATCHING_WEIGHTS.
    
    Args:
        df: DataFrame with student data
    
    Returns:
        tuple: (feature_matrix, student_names, feature_weights)
    """
    try:
        # Initialize label encoders for categorical fields
        label_encoders = {}
        for field in CATEGORICAL_FIELDS:
            if field in df.columns:
                label_encoders[field] = LabelEncoder()
                # Get unique values, handling None/NaN
                unique_vals = df[field].dropna().astype(str).unique()
                if len(unique_vals) > 0:
                    label_encoders[field].fit(unique_vals)
        
        # Prepare feature matrix
        features = []
        student_names = []
        feature_weights = []
        
        # Get all unique values for multi-value fields
        all_interests = set()
        all_cuisines = set()
        all_movies = set()
        
        for field in MULTI_VALUE_FIELDS:
            if field in df.columns:
                for val in df[field]:
                    if isinstance(val, list):
                        if field == 'interests':
                            all_interests.update(val)
                        elif field == 'cuisine':
                            all_cuisines.update(val)
                        elif field == 'movies':
                            all_movies.update(val)
        
        # Create feature vectors for each student
        for idx, row in df.iterrows():
            try:
                # Get student name
                student_name = row.get('name') if hasattr(row, 'get') else row['name']
                if _safe_is_none_or_nan(student_name):
                    continue
                student_name = str(student_name).strip()
                if not student_name:
                    continue
                
                student_names.append(student_name)
                student_features = []
                weights_list = []
                
                # Encode categorical fields with weights
                for field in CATEGORICAL_FIELDS:
                    if field in df.columns and field in MATCHING_WEIGHTS:
                        weight = MATCHING_WEIGHTS[field]
                        val = row[field]
                        
                        # Handle None/NaN
                        if _safe_is_none_or_nan(val):
                            student_features.append(0.0)
                            weights_list.append(weight)
                        else:
                            # Encode categorical value
                            try:
                                encoded = label_encoders[field].transform([str(val)])[0]
                                # Normalize to 0-1 range
                                n_classes = len(label_encoders[field].classes_)
                                normalized = encoded / max(n_classes - 1, 1) if n_classes > 1 else 0.0
                                student_features.append(normalized)
                                weights_list.append(weight)
                            except (ValueError, KeyError):
                                student_features.append(0.0)
                                weights_list.append(weight)
                
                # Encode multi-value fields with Jaccard Index representation
                for field in MULTI_VALUE_FIELDS:
                    if field in df.columns and field in MATCHING_WEIGHTS:
                        weight = MATCHING_WEIGHTS[field]
                        val = row[field]
                        
                        # Get all unique values for this field
                        if field == 'interests':
                            all_values = all_interests
                        elif field == 'cuisine':
                            all_values = all_cuisines
                        elif field == 'movies':
                            all_values = all_movies
                        else:
                            all_values = set()
                        
                        # Create binary vector for this field
                        if _safe_is_none_or_nan(val):
                            # All zeros
                            for _ in all_values:
                                student_features.append(0.0)
                                weights_list.append(weight / max(len(all_values), 1))
                        else:
                            # Convert to set
                            if isinstance(val, list):
                                val_set = set(val)
                            elif isinstance(val, str) and val.strip():
                                try:
                                    parsed = ast.literal_eval(val)
                                    val_set = set(parsed) if isinstance(parsed, list) else set()
                                except (ValueError, SyntaxError):
                                    val_set = set()
                            else:
                                val_set = set()
                            
                            # Create binary features for each unique value
                            for unique_val in all_values:
                                if unique_val in val_set:
                                    student_features.append(1.0)
                                else:
                                    student_features.append(0.0)
                                weights_list.append(weight / max(len(all_values), 1))
                
                features.append(student_features)
                
            except Exception as e:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Error preparing features for student at index {idx}: {e}")
                continue
        
        if len(features) == 0:
            return None, None, None
        
        # Convert to numpy array
        feature_matrix = np.array(features)
        weights_array = np.array(weights_list[:len(features[0])])
        
        # Apply weights to features (multiply each feature by its weight)
        weighted_features = feature_matrix * weights_array
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(weighted_features)
        
        return scaled_features, student_names, weights_array
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error preparing features: {e}")
            import traceback
            traceback.print_exc()
        return None, None, None


def _determine_optimal_k(feature_matrix, min_k=2, max_k=20):
    """
    Determine optimal number of clusters using elbow method and silhouette score.
    
    Args:
        feature_matrix: Feature matrix for clustering
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters
    
    Returns:
        int: Optimal number of clusters
    """
    try:
        if len(feature_matrix) < min_k:
            return min(len(feature_matrix), min_k)
        
        max_k = min(max_k, len(feature_matrix) - 1)
        if max_k < min_k:
            return min_k
        
        best_k = min_k
        best_silhouette = -1
        
        # Try different values of k
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                labels = kmeans.fit_predict(feature_matrix)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(feature_matrix, labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
            except Exception as e:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Error evaluating k={k}: {e}")
                continue
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Optimal k={best_k} (silhouette score: {best_silhouette:.3f})")
        
        return best_k
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error determining optimal k: {e}")
        return min_k


def _kmeans_clustering_matching(df):
    """
    Match students using K-means clustering with improved preprocessing pipeline.
    Uses one-hot encoding, multi-label encoding, and Elbow method for optimal K.
    
    Args:
        df: DataFrame with student data
    
    Returns:
        dict: Dictionary mapping student names to their top matches
    """
    try:
        # Preprocess data using new pipeline
        preprocessed_df, preprocessing_info = preprocess_data(df)
        
        if preprocessed_df is None or len(preprocessed_df) == 0:
            if ENABLE_DEBUG_LOGGING:
                print("⚠️ Could not preprocess data for clustering, falling back to weighted similarity")
            return _weighted_similarity_matching(df)
        
        # Run K-Means with optimal K determined by Elbow method
        labels, kmeans_model = run_kmeans(preprocessed_df, n_clusters=None)
        
        if labels is None:
            if ENABLE_DEBUG_LOGGING:
                print("⚠️ Could not run K-Means clustering, falling back to weighted similarity")
            return _weighted_similarity_matching(df)
        
        # Prepare output with cluster labels attached
        output_df = prepare_output(df, labels, preprocessing_info)
        
        # Create cluster assignments dictionary
        cluster_assignments = {}
        for idx, row in output_df.iterrows():
            student_name = row.get('name') if 'name' in row else None
            cluster_id = row.get('cluster') if 'cluster' in row else -1
            if student_name:
                cluster_assignments[str(student_name).strip()] = int(cluster_id)
        
        # Group students by cluster
        cluster_groups = {}
        for student_name, cluster_id in cluster_assignments.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(student_name)
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Created {len(cluster_groups)} clusters:")
            for cluster_id, students in sorted(cluster_groups.items()):
                print(f"  Cluster {cluster_id}: {len(students)} students")
        
        # Generate matches for each student
        matches = {}
        
        # Normalize column names in original df
        column_mapping = {
            'Name': 'name',
            'Email': 'email',
            'Major': 'major',
            'Year': 'year',
            'Language': 'language',
            'Country': 'country',
            'Personality': 'personality',
            'Study Style': 'studyStyle',
            'Cuisine': 'cuisine',
            'Interests': 'interests',
            'Movies': 'movies'
        }
        df_normalized = df.copy().rename(columns=column_mapping)
        
        # Pre-compute student data dictionary for performance optimization
        student_data_dict = {}
        for idx, row in df_normalized.iterrows():
            try:
                name = str(row.get('name', '')).strip()
                if name:
                    student_data_dict[name] = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            except (KeyError, TypeError, AttributeError):
                continue
        
        for student_name in cluster_assignments.keys():
            try:
                student_matches = []
                student_cluster = cluster_assignments[student_name]
                
                # Get student data using optimized lookup
                student_data = student_data_dict.get(student_name)
                if student_data is None:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Student '{student_name}' not found")
                    continue
                
                # Get students in the same cluster
                same_cluster_students = [s for s in cluster_groups[student_cluster] if s != student_name]
                
                # Calculate similarity with students in same cluster
                dynamic_threshold = _get_dynamic_threshold(len(df_normalized))
                for match_name in same_cluster_students:
                    try:
                        match_data = student_data_dict.get(match_name)
                        if match_data is None:
                            continue
                        
                        similarity = calculate_weighted_similarity(student_data, match_data)
                        
                        # Apply cluster boost: same-cluster matches get 5% boost
                        # This rewards K-means clustering results while maintaining score integrity
                        cluster_boost = 5.0  # Same cluster = True
                        boosted_similarity = min(100.0, similarity + cluster_boost)
                        
                        if boosted_similarity >= dynamic_threshold:
                            student_matches.append((match_name, boosted_similarity, True))  # True = same cluster
                    except Exception as e:
                        if ENABLE_DEBUG_LOGGING:
                            print(f"⚠️ Error calculating similarity for {student_name} -> {match_name}: {e}")
                        continue
                
                # If enabled, also check students in other clusters
                if ENABLE_CROSS_CLUSTER_MATCHING:
                    for cluster_id, cluster_students in cluster_groups.items():
                        if cluster_id != student_cluster:
                            for match_name in cluster_students:
                                try:
                                    match_data = student_data_dict.get(match_name)
                                    if match_data is None:
                                        continue
                                    
                                    similarity = calculate_weighted_similarity(student_data, match_data)
                                    
                                    if similarity >= dynamic_threshold:
                                        student_matches.append((match_name, similarity, False))  # False = different cluster
                                except Exception as e:
                                    if ENABLE_DEBUG_LOGGING:
                                        print(f"⚠️ Error calculating similarity for {student_name} -> {match_name}: {e}")
                                    continue
                
                # Normalize scores to 0-100 range based on actual distribution
                # This makes matches look better: if max is 33%, it becomes 100%
                student_matches = _normalize_scores(student_matches, student_data, student_data_dict)
                
                # Enforce diversity in matches
                if ENABLE_DIVERSITY_ENFORCEMENT:
                    student_matches = enforce_diversity_in_matches(student_matches, student_data, student_data_dict)
                
                # Sort matches: same cluster first, then by normalized similarity score
                student_matches.sort(key=lambda x: (not x[2], -x[1]))  # Same cluster first, then by similarity
                
                # Limit matches from same cluster
                same_cluster_matches = [m for m in student_matches if m[2]][:MAX_MATCHES_FROM_SAME_CLUSTER]
                other_matches = [m for m in student_matches if not m[2]]
                
                # Combine and take top N
                top_matches = same_cluster_matches + other_matches
                top_matches = top_matches[:TOP_MATCHES_COUNT]
                
                # FALLBACK: If no matches found, retry with 50% lower threshold
                if len(top_matches) == 0:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ No matches for {student_name} at {dynamic_threshold}% threshold, retrying with lower threshold...")
                    
                    fallback_threshold = dynamic_threshold * 0.5  # 50% lower (e.g., 38% -> 19%)
                    fallback_matches = []
                    
                    for match_name, match_data in student_data_dict.items():
                        if match_name != student_name:
                            try:
                                similarity = calculate_weighted_similarity(student_data, match_data)
                                if similarity >= fallback_threshold:
                                    fallback_matches.append((match_name, similarity, False))
                            except Exception as e:
                                if ENABLE_DEBUG_LOGGING:
                                    print(f"⚠️ Error in fallback matching: {e}")
                                continue
                    
                    # Sort by similarity and take top matches
                    fallback_matches.sort(key=lambda x: -x[1])
                    top_matches = fallback_matches[:TOP_MATCHES_COUNT]
                    
                    if ENABLE_DEBUG_LOGGING:
                        print(f"✓ Found {len(top_matches)} matches for {student_name} with fallback threshold ({fallback_threshold}%)")
                
                # Convert to detailed match format
                detailed_matches = []
                for match_name, similarity_score, _ in top_matches:
                    try:
                        match_info = _create_detailed_match(student_data_dict, student_name, match_name, similarity_score)
                        detailed_matches.append(match_info)
                    except Exception as e:
                        if ENABLE_DEBUG_LOGGING:
                            print(f"⚠️ Error creating detailed match for {student_name} -> {match_name}: {e}")
                        continue
                
                matches[student_name] = detailed_matches
                
            except Exception as e:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Error processing student {student_name}: {e}")
                continue
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Generated K-means clustering matches for {len(matches)} students")
        
        return matches
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error in K-means clustering matching: {e}")
            import traceback
            traceback.print_exc()
        # Fall back to weighted similarity matching
        return _weighted_similarity_matching(df)


def _hybrid_clustering_matching(df):
    """
    Hybrid approach: K-means clustering + weighted similarity matching within clusters.
    This combines the benefits of both approaches.
    
    Args:
        df: DataFrame with student data
    
    Returns:
        dict: Dictionary mapping student names to their top matches
    """
    try:
        # First, perform K-means clustering to group students
        feature_matrix, student_names, feature_weights = _prepare_features_for_clustering(df)
        
        if feature_matrix is None or len(feature_matrix) == 0:
            if ENABLE_DEBUG_LOGGING:
                print("⚠️ Could not prepare features for clustering, using weighted similarity only")
            return _weighted_similarity_matching(df)
        
        # Determine optimal number of clusters
        if KMEANS_N_CLUSTERS is None:
            n_clusters = _determine_optimal_k(feature_matrix, KMEANS_MIN_CLUSTERS, KMEANS_MAX_CLUSTERS)
        else:
            n_clusters = min(KMEANS_N_CLUSTERS, len(feature_matrix))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Create cluster assignments
        cluster_assignments = {}
        for i, student_name in enumerate(student_names):
            cluster_assignments[student_name] = cluster_labels[i]
        
        # Generate matches using weighted similarity, but prioritize within-cluster matches
        matches = {}
        
        for student_name in student_names:
            student_cluster = cluster_assignments[student_name]
            # Safely get student data
            student_filter = df[df['name'] == student_name]
            if len(student_filter) == 0:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Student '{student_name}' not found in DataFrame")
                continue
            student_data = student_filter.iloc[0].to_dict()
            student_matches = []
            
            # Calculate similarity with all other students
            for match_name in student_names:
                if match_name == student_name:
                    continue
                
                try:
                    match_filter = df[df['name'] == match_name]
                    if len(match_filter) == 0:
                        continue
                    match_data = match_filter.iloc[0].to_dict()
                    similarity = calculate_weighted_similarity(student_data, match_data)
                    is_same_cluster = cluster_assignments[match_name] == student_cluster
                    
                    # Apply cluster boost: same-cluster matches get 5% boost
                    if is_same_cluster:
                        similarity = min(100.0, similarity + 5.0)  # 5% boost
                    
                    # Use dynamic threshold based on number of students
                    dynamic_threshold = _get_dynamic_threshold(len(df))
                    
                    if similarity >= dynamic_threshold:
                        student_matches.append((match_name, similarity, is_same_cluster))
                except Exception as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Error calculating similarity for {student_name} -> {match_name}: {e}")
                    continue
            
            # Normalize scores to 0-100 range based on actual distribution
            student_matches = _normalize_scores(student_matches, student_data, df)
            
            # Enforce diversity in matches
            if ENABLE_DIVERSITY_ENFORCEMENT:
                student_matches = enforce_diversity_in_matches(student_matches, student_data, df)
            
            # Sort matches: same cluster first (with boost), then by normalized similarity
            student_matches.sort(key=lambda x: (not x[2], -x[1]))
            
            # Take top N matches
            top_matches = student_matches[:TOP_MATCHES_COUNT]
            
            # Convert to detailed match format
            detailed_matches = []
            for match_name, similarity_score, _ in top_matches:
                try:
                    match_info = _create_detailed_match(df, student_name, match_name, similarity_score)
                    detailed_matches.append(match_info)
                except Exception as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Error creating detailed match for {student_name} -> {match_name}: {e}")
                    continue
            
            matches[student_name] = detailed_matches
        
        if ENABLE_DEBUG_LOGGING:
            print(f"Generated hybrid clustering matches for {len(matches)} students")
        
        return matches
        
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error in hybrid clustering matching: {e}")
            import traceback
            traceback.print_exc()
        # Fall back to weighted similarity matching
        return _weighted_similarity_matching(df)


def _fallback_similarity_matching(df):
    """
    Fallback matching using simple similarity scoring when weighted matching fails.
    """
    try:
        # Set random seed for reproducibility if specified
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        
        matches = {}
        
        for i, student1 in df.iterrows():
            # Get student1 name safely, skip if None/NaN
            try:
                student1_name = student1.get('name', None) if hasattr(student1, 'get') else student1['name']
                if student1_name is None:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Skipping student at index {i}: name is None")
                    continue
                try:
                    if pd.isna(student1_name):
                        if ENABLE_DEBUG_LOGGING:
                            print(f"⚠️ Skipping student at index {i}: name is NaN")
                        continue
                except (TypeError, ValueError):
                    pass
                student1_name = str(student1_name).strip()
                if not student1_name:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Skipping student at index {i}: name is empty")
                    continue
            except (KeyError, TypeError, AttributeError) as e:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Skipping student at index {i}: error getting name: {e}")
                continue
            
            # Get student1 data as dict for diversity calculations
            student1_data = student1.to_dict() if hasattr(student1, 'to_dict') else dict(student1)
            
            student_matches = []
            
            for j, student2 in df.iterrows():
                if i != j:  # Don't match with self
                    # Get student2 name safely, skip if None/NaN
                    try:
                        student2_name = student2.get('name') if hasattr(student2, 'get') else student2['name']
                        # Skip None/NaN values
                        if student2_name is None:
                            continue
                        try:
                            if pd.isna(student2_name):
                                continue
                        except (TypeError, ValueError):
                            pass
                        # Convert to string and validate
                        student2_name = str(student2_name).strip()
                        if not student2_name:
                            continue
                    except (KeyError, TypeError, AttributeError):
                        continue
                    
                    score = _calculate_simple_similarity(student1, student2)
                    # Use dynamic threshold based on number of students
                    dynamic_threshold = _get_dynamic_threshold(len(df))
                    if score >= dynamic_threshold:
                        student_matches.append((student2_name, score))
            
            # Normalize scores to 0-100 range based on actual distribution
            # Convert to format expected by _normalize_scores (add is_same_cluster=False)
            if student_matches:
                student_matches_with_cluster = [(name, score, False) for name, score in student_matches]
                normalized_matches = _normalize_scores(student_matches_with_cluster, student1_data, df)
                student_matches = [(name, score) for name, score, _ in normalized_matches]
            
            # Sort by normalized score and take top N matches
            student_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Enforce diversity in matches
            if ENABLE_DIVERSITY_ENFORCEMENT:
                student_matches_with_cluster = [(name, score, False) for name, score in student_matches]
                student_matches_with_cluster = enforce_diversity_in_matches(
                    student_matches_with_cluster, student1_data, df
                )
                student_matches = [(name, score) for name, score, _ in student_matches_with_cluster]
            
            top_matches = student_matches[:TOP_MATCHES_COUNT]
            
            # Fair tie-breaking if enabled
            if ENABLE_TIE_BREAKING and len(top_matches) > 1:
                random.shuffle(top_matches)
            
            # Convert to detailed match format
            detailed_matches = []
            for match_name, similarity_score in top_matches:
                try:
                    match_info = _create_detailed_match(df, student1_name, match_name, similarity_score)
                    detailed_matches.append(match_info)
                except Exception as e:
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Error creating detailed match for {student1_name} -> {match_name}: {e}")
                    continue
            
            matches[student1_name] = detailed_matches
        
        return matches
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error in fallback matching: {e}")
        return {}


def _calculate_simple_similarity(student1, student2):
    """
    Calculate simple similarity score between two students (fallback method).
    """
    score = 0.0
    weights = FALLBACK_WEIGHTS
    total_weight = sum(weights.values())
    
    # Exact matches for categorical fields
    for field in CATEGORICAL_FIELDS:
        if field in student1 and field in student2 and field in weights:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Check for exact match and non-empty values
            if val1 == val2 and val1:
                score += weights[field]
    
    # List overlap for multi-value fields
    for field in MULTI_VALUE_FIELDS:
        if field in student1 and field in student2 and field in weights:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Convert to lists safely
            if isinstance(val1, list):
                list1 = val1
            elif isinstance(val1, str) and val1.strip():
                # Handle string representations of lists
                try:
                    parsed = ast.literal_eval(val1)
                    list1 = parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    list1 = []
            else:
                list1 = []
            
            if isinstance(val2, list):
                list2 = val2
            elif isinstance(val2, str) and val2.strip():
                # Handle string representations of lists
                try:
                    parsed = ast.literal_eval(val2)
                    list2 = parsed if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    list2 = []
            else:
                list2 = []
            
            if list1 and list2:
                overlap = len(set(list1) & set(list2))
                max_len = max(len(list1), len(list2))
                if max_len > 0:
                    score += (overlap / max_len) * weights[field]
    
    # Convert to percentage (0-100)
    similarity_percentage = (score / total_weight) * 100 if total_weight > 0 else 0.0
    
    return round(similarity_percentage, SIMILARITY_SCORE_PRECISION)


def _create_detailed_match(df_or_dict, student_name, match_name, similarity_score):
    """
    Create detailed match information including explanations and commonalities.
    
    Args:
        df_or_dict: DataFrame with all student data OR dictionary mapping names to student data
        student_name: Name of the student looking for matches
        match_name: Name of the matched student
        similarity_score: Similarity score between the students
    
    Returns:
        dict: Detailed match information
    """
    try:
        # Handle both DataFrame and dictionary inputs for backward compatibility
        if isinstance(df_or_dict, dict):
            student_data_dict = df_or_dict
            student_data = student_data_dict.get(student_name)
            match_data = student_data_dict.get(match_name)
        else:
            df = df_or_dict
            # Get student data - check if students exist before accessing
            # Use normalized string comparison for consistency with rest of codebase
            student_name_normalized = str(student_name).strip()
            match_name_normalized = str(match_name).strip()
            student_filter = df[df['name'].astype(str).str.strip() == student_name_normalized]
            match_filter = df[df['name'].astype(str).str.strip() == match_name_normalized]
            
            # Check if students exist
            if len(student_filter) == 0:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Student '{student_name}' not found in DataFrame")
                raise ValueError(f"Student '{student_name}' not found")
            
            if len(match_filter) == 0:
                if ENABLE_DEBUG_LOGGING:
                    print(f"⚠️ Match '{match_name}' not found in DataFrame")
                raise ValueError(f"Match '{match_name}' not found")
            
            # Get student data as Series and convert to dict for safe access
            student_data = student_filter.iloc[0].to_dict()
            match_data = match_filter.iloc[0].to_dict()
        
        # Check if students exist (for dictionary case)
        if student_data is None:
            if ENABLE_DEBUG_LOGGING:
                print(f"⚠️ Student '{student_name}' not found")
            raise ValueError(f"Student '{student_name}' not found")
        
        if match_data is None:
            if ENABLE_DEBUG_LOGGING:
                print(f"⚠️ Match '{match_name}' not found")
            raise ValueError(f"Match '{match_name}' not found")
        
        # Calculate commonalities
        commonalities = _find_commonalities(student_data, match_data)
        
        # Generate explanation
        explanation = _generate_explanation(student_data, match_data, commonalities, similarity_score)
        
        # Similarity score is already a percentage from weighted calculation
        similarity_percentage = round(similarity_score, FINAL_SCORE_PRECISION)
        
        # Helper function to safely get value from dict
        def safe_get(data, key, default):
            value = data.get(key, default)
            # Handle NaN/None values from pandas
            if value is None:
                return default
            # Check if value is NaN (pandas uses NaN for missing values)
            try:
                if pd.isna(value):
                    return default
            except (TypeError, ValueError):
                # Value is not NaN-compatible type, use as-is
                pass
            return value
        
        return {
            'name': match_name,
            'email': safe_get(match_data, 'email', ''),
            'similarity_score': similarity_percentage,
            'explanation': explanation,
            'commonalities': commonalities,
            'profile': {
                'major': safe_get(match_data, 'major', ''),
                'year': safe_get(match_data, 'year', ''),
                'language': safe_get(match_data, 'language', ''),
                'country': safe_get(match_data, 'country', ''),
                'personality': safe_get(match_data, 'personality', ''),
                'studyStyle': safe_get(match_data, 'studyStyle', ''),
                'cuisine': safe_get(match_data, 'cuisine', []),
                'interests': safe_get(match_data, 'interests', []),
                'movies': safe_get(match_data, 'movies', [])
            }
        }
    except Exception as e:
        if ENABLE_DEBUG_LOGGING:
            print(f"Error creating detailed match: {e}")
            import traceback
            traceback.print_exc()
        return {
            'name': match_name,
            'email': '',
            'similarity_score': 0,
            'explanation': 'Match found but details unavailable',
            'commonalities': [],
            'profile': {}
        }


def _find_commonalities(student1, student2):
    """
    Find commonalities between two students using Jaccard Index for multi-value fields.
    
    Args:
        student1, student2: Student data dictionaries
    
    Returns:
        list: List of commonalities
    """
    commonalities = []
    
    # Categorical matches (exact matches)
    for field in CATEGORICAL_FIELDS:
        if field in student1 and field in student2:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Check for exact match and non-empty values
            if val1 == val2 and val1:
                try:
                    if field == 'major':
                        commonalities.append(f"Both studying {val1}")
                    elif field == 'year':
                        commonalities.append(f"Both in {val1}")
                    elif field == 'language':
                        commonalities.append(f"Both speak {val1}")
                    elif field == 'country':
                        commonalities.append(f"Both from {val1}")
                    elif field == 'personality':
                        commonalities.append(f"Both are {str(val1).lower()}")
                    elif field == 'studyStyle':
                        commonalities.append(f"Both prefer {str(val1).lower()}")
                except (AttributeError, TypeError) as e:
                    # Skip if value can't be converted to string
                    if ENABLE_DEBUG_LOGGING:
                        print(f"⚠️ Error processing commonality for {field}: {e}")
                    continue
    
    # Multi-value fields using Jaccard Index
    for field in MULTI_VALUE_FIELDS:
        if field in student1 and field in student2:
            val1 = student1[field]
            val2 = student2[field]
            # Skip None/NaN values
            if val1 is None or val2 is None:
                continue
            try:
                if pd.isna(val1) or pd.isna(val2):
                    continue
            except (TypeError, ValueError):
                pass
            # Convert to sets safely
            if isinstance(val1, list):
                list1 = set(val1)
            elif isinstance(val1, str) and val1.strip():
                # Handle string representations of lists
                try:
                    parsed = ast.literal_eval(val1)
                    list1 = set(parsed) if isinstance(parsed, list) else set()
                except (ValueError, SyntaxError):
                    list1 = set()
            else:
                list1 = set()
            
            if isinstance(val2, list):
                list2 = set(val2)
            elif isinstance(val2, str) and val2.strip():
                # Handle string representations of lists
                try:
                    parsed = ast.literal_eval(val2)
                    list2 = set(parsed) if isinstance(parsed, list) else set()
                except (ValueError, SyntaxError):
                    list2 = set()
            else:
                list2 = set()
            
            if list1 and list2:
                common_items = list1 & list2
                if common_items:
                    try:
                        if field == 'cuisine':
                            if len(common_items) == 1:
                                commonalities.append(f"Both enjoy {list(common_items)[0]} cuisine")
                            else:
                                commonalities.append(f"Both enjoy {', '.join(list(common_items)[:2])} cuisine")
                        elif field == 'interests':
                            if len(common_items) == 1:
                                commonalities.append(f"Both interested in {list(common_items)[0]}")
                            else:
                                commonalities.append(f"Both interested in {', '.join(list(common_items)[:2])}")
                        elif field == 'movies':
                            if len(common_items) == 1:
                                commonalities.append(f"Both like {list(common_items)[0]} movies")
                            else:
                                commonalities.append(f"Both like {', '.join(list(common_items)[:2])} movies")
                    except (TypeError, AttributeError) as e:
                        # Skip if items can't be converted to string
                        if ENABLE_DEBUG_LOGGING:
                            print(f"⚠️ Error processing commonality for {field}: {e}")
                        continue
    
    return commonalities


def _generate_explanation(student1, student2, commonalities, similarity_score):
    """
    Generate a human-readable explanation for why two students match.
    
    Args:
        student1, student2: Student data dictionaries
        commonalities: List of commonalities
        similarity_score: Similarity score (0-100 percentage)
    
    Returns:
        str: Explanation text
    """
    # Use thresholds from configuration (similarity_score is 0-100, not 0-1)
    thresholds = SIMILARITY_THRESHOLDS
    if similarity_score >= thresholds['excellent']:
        strength = "excellent"
    elif similarity_score >= thresholds['very_good']:
        strength = "very good"
    elif similarity_score >= thresholds['good']:
        strength = "good"
    else:
        strength = "decent"
    
    if not commonalities:
        return f"You have a {strength} compatibility match! While you may have different backgrounds, your profiles suggest you could complement each other well."
    
    if len(commonalities) == 1:
        return f"You have a {strength} compatibility match! {commonalities[0]}, which suggests you'd get along well."
    elif len(commonalities) == 2:
        return f"You have a {strength} compatibility match! {commonalities[0]} and {commonalities[1]}, making you great study partners."
    else:
        main_commonalities = commonalities[:2]
        return f"You have a {strength} compatibility match! {', '.join(main_commonalities)}, and more shared interests, making you perfect study buddies!"
