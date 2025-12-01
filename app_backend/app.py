from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from matcher import generate_matches
import requests
from config import MIN_SIMILARITY_SCORE

app = Flask(__name__)
CORS(app)

NODE_API = "http://localhost:3001/api/students"

def load_students_data(max_retries=3, retry_delay=2):
    """Load student data from Node.js API with retry logic"""
    for attempt in range(max_retries):
        try:
            # Use localhost consistently (same as Node.js uses for Flask)
            response = requests.get(NODE_API, timeout=15)
            response.raise_for_status()
            students = response.json()
            print(f"[OK] Loaded {len(students)} students from Node.js API")
            
            # Debug: Print first student to check data format
            if students:
                print(f"DEBUG: First student sample: {students[0]}")
                print(f"DEBUG: First student columns: {list(students[0].keys())}")
            else:
                print("[WARN] Warning: Node.js API returned empty list")
            
            return students
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Connection error (attempt {attempt + 1}/{max_retries}): Node.js server not running on port 3001")
                print(f"   Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] Connection error: Node.js server not running on port 3001")
                print(f"   Error: {e}")
                print(f"   Make sure Node.js server is running: npm start")
                return []
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Timeout error (attempt {attempt + 1}/{max_retries}): Node.js API took too long to respond")
                print(f"   Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] Timeout error: Node.js API took too long to respond")
                print(f"   Error: {e}")
                return []
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error: Node.js API returned error status")
            print(f"   Status: {e.response.status_code}")
            print(f"   Error: {e}")
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Error loading from API (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] Could not load from API: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                return []
    
    return []

@app.route('/')
def home():
    return jsonify({"message": "Student AI Matcher backend is running!"})

@app.route('/api/match', methods=['GET'])
def match_students():
    try:
        # Load students from API
        students = load_students_data()

        if not students:
            error_msg = "No student data found. Please check if Node.js server is running on port 3001 and Google Sheets has data."
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "error": error_msg,
                "success": False
            }), 404

        df = pd.DataFrame(students)
        print(f"[INFO] Loaded {len(df)} students from Node.js API")
        print(f"[INFO] DataFrame shape: {df.shape}")
        print(f"[INFO] DataFrame columns: {list(df.columns)}")
        
        # Check if name column exists
        name_col = 'name' if 'name' in df.columns else 'Name'
        if name_col not in df.columns:
            print(f"[ERROR] Name column not found. Available columns: {list(df.columns)}")
            return jsonify({
                "error": "Name column not found in student data",
                "success": False
            }), 500
        
        results = generate_matches(df)
        print(f"[INFO] Generated matches for {len(results)} students")
        print(f"[INFO] Total students in database: {len(df)}")
        if len(results) > 0:
            print(f"[INFO] Results keys (student names): {list(results.keys())[:10]}")  # Show first 10
            # Debug: Print structure of first result
            first_key = list(results.keys())[0]
            if results[first_key] and len(results[first_key]) > 0:
                print(f"[INFO] First match example keys: {list(results[first_key][0].keys())}")
                print(f"[INFO] First match sample: name={results[first_key][0].get('name')}, email={results[first_key][0].get('email')}, score={results[first_key][0].get('similarity_score')}")
            # Print match counts for all students
            print(f"[INFO] Match counts per student:")
            for key, value in list(results.items())[:10]:
                match_count = len(value) if value else 0
                print(f"   {key}: {match_count} matches")
        else:
            print(f"[WARN] No matches generated - results dictionary is empty")
            print(f"   This might be because:")
            print(f"   - There are fewer than 2 students in the database")
            print(f"   - All similarity scores are below MIN_SIMILARITY_SCORE (currently {MIN_SIMILARITY_SCORE})")

        # [OK] Check if user name is passed as query param
        user_name = request.args.get('name', '').strip()
        print(f"[SEARCH] Looking for matches for user: '{user_name}'")

        if user_name:
            # Check if student exists in the dataframe (case-insensitive)
            # Try to find student by name (case-insensitive match)
            student_found = False
            actual_student_name = None
            
            for idx, row in df.iterrows():
                # Safely get name value, handling None/NaN
                name_value = row[name_col]
                # Skip None/NaN values
                if name_value is None:
                    continue
                try:
                    if pd.isna(name_value):
                        continue
                except (TypeError, ValueError):
                    pass
                # Convert to string and check match
                try:
                    name_str = str(name_value).strip()
                    if name_str and name_str.lower() == user_name.lower():
                        student_found = True
                        actual_student_name = name_str
                        print(f"[OK] Found student: '{actual_student_name}' (matched '{user_name}')")
                        break
                except (TypeError, AttributeError):
                    # Skip if value can't be converted to string
                    continue
            
            if not student_found:
                print(f"[WARN] Student '{user_name}' not found in database")
                print(f"   Total students in database: {len(df)}")
                print(f"   Available students: {df[name_col].tolist()[:10]}...")
                # Student might not be in database yet, return empty matches
                # This can happen if Google Sheets hasn't updated yet
                return jsonify({
                    "success": True,
                    "student_name": user_name,
                    "matches": [],
                    "total_matches": 0,
                    "message": "Student not found in database yet. Please try again in a moment."
                })
            
            # Use the actual student name from the database (might have different casing)
            search_name = actual_student_name if actual_student_name else user_name
            
            # Check if matches exist for this student
            # Try exact match first using the actual student name from database
            matches = results.get(search_name, [])
            
            print(f"[SEARCH] Checking matches for '{search_name}'")
            print(f"   Exact match result: {matches}")
            print(f"   Results keys available: {list(results.keys())[:10]}")
            
            # If no exact match, try case-insensitive match
            if not matches:
                print(f"   No exact match found, trying case-insensitive match...")
                for key in results.keys():
                    try:
                        # Safely convert key to string, handling None/NaN
                        if key is None:
                            continue
                        try:
                            if pd.isna(key):
                                continue
                        except (TypeError, ValueError):
                            pass
                        key_str = str(key).strip() if key else ''
                        if key_str and key_str.lower() == search_name.lower():
                            matches = results[key]
                            print(f"[OK] Found matches using case-insensitive match: '{key}' -> {len(matches)} matches")
                            break
                    except (TypeError, AttributeError):
                        # Skip if key can't be converted to string
                        continue
                if not matches:
                    print(f"[WARN] No matches found in results dictionary for '{search_name}'")
                    print(f"   Available keys: {list(results.keys())}")
            
            match_count = len(matches) if matches else 0
            print(f"[INFO] Found {match_count} matches for '{search_name}'")
            print(f"   Student searched: '{user_name}'")
            print(f"   Student found in DB: '{search_name}'")
            if match_count > 0:
                print(f"   Match names: {[m.get('name', 'N/A') if isinstance(m, dict) else str(m) for m in matches]}")
            
            if match_count > 0:
                print(f"[OK] Returning {match_count} matches for {search_name}")
                print(f"   Match names: {[m.get('name', 'N/A') for m in matches]}")
                
                # Debug: Print first match structure
                if matches and len(matches) > 0:
                    print(f"   First match structure: {list(matches[0].keys())}")
                    print(f"   First match sample: name={matches[0].get('name')}, email={matches[0].get('email')}, score={matches[0].get('similarity_score')}")
                
                # Ensure matches are properly formatted for JSON serialization
                formatted_matches = []
                for match in matches:
                    if isinstance(match, dict):
                        # Ensure all values are JSON-serializable
                        # Format profile to ensure JSON serialization
                        profile = match.get('profile', {})
                        formatted_profile = {}
                        if isinstance(profile, dict):
                            for key, value in profile.items():
                                if isinstance(value, (list, tuple)):
                                    formatted_profile[key] = [str(v) for v in value]
                                elif value is None or (isinstance(value, float) and pd.isna(value)):
                                    formatted_profile[key] = ''
                                else:
                                    formatted_profile[key] = str(value) if value is not None else ''
                        
                        formatted_match = {
                            'name': str(match.get('name', '')),
                            'email': str(match.get('email', '')),
                            'similarity_score': float(match.get('similarity_score', 0)),
                            'explanation': str(match.get('explanation', '')),
                            'commonalities': [str(c) for c in match.get('commonalities', [])] if isinstance(match.get('commonalities'), (list, tuple)) else [],
                            'profile': formatted_profile
                        }
                        formatted_matches.append(formatted_match)
                
                # Final validation: Ensure we have valid matches
                if len(formatted_matches) == 0:
                    print(f"[WARN] Warning: No valid formatted matches found for {search_name}")
                    print(f"   Original matches count: {len(matches)}")
                    print(f"   Match types: {[type(m).__name__ for m in matches[:3]]}")
                else:
                    print(f"[OK] Formatted {len(formatted_matches)} valid matches for {search_name}")
                
                response_data = {
                    "success": True,
                    "student_name": search_name,
                    "matches": formatted_matches,
                    "total_matches": len(formatted_matches)
                }
                
                print(f"   Returning response with {len(formatted_matches)} formatted matches")
                
                # Final validation before returning
                if len(formatted_matches) > 0:
                    print(f"   First match in response: name={formatted_matches[0].get('name')}, email={formatted_matches[0].get('email')}, score={formatted_matches[0].get('similarity_score')}")
                else:
                    print(f"   [WARN] WARNING: Returning empty matches array for {search_name}")
                    print(f"      This means no matches were found or all matches were filtered out")
                
                return jsonify(response_data)
            else:
                # No matches found (likely first student or no compatible matches)
                print(f"[WARN] No matches found for {search_name}")
                print(f"   Total students in database: {len(df)}")
                print(f"   Students with matches: {len(results)}")
                if len(results) > 0:
                    print(f"   Students who have matches: {list(results.keys())[:5]}")
                # Return empty matches but still with success=True so email is sent
                empty_response = {
                    "success": True,
                    "student_name": search_name,
                    "matches": [],  # Explicitly empty array
                    "total_matches": 0,
                    "message": "No matches available yet. More students needed for matching."
                }
                print(f"   Returning empty matches response for {search_name}")
                return jsonify(empty_response)

        print("Returning all matches")
        return jsonify({
            "success": True,
            "all_matches": results,
            "total_students": len(results)
        })

    except Exception as e:
        print(f"[ERROR] Error in match_students: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/match_for_student', methods=['POST'])
def get_matches_for_student():
    """Get matches for a specific student"""
    try:
        student_data = request.get_json()
        student_name = student_data.get('name', '')
        
        # Get all students from Node.js API
        students = load_students_data()
        
        if not students:
            return jsonify({"error": "No student data available"}), 404
        
        # Generate matches
        df = pd.DataFrame(students)
        results = generate_matches(df)
        
        # Return matches for this student
        matches = results.get(student_name, [])
        
        return jsonify({
            "success": True,
            "student_name": student_name,
            "matches": matches,
            "total_matches": len(matches)
        })
    except Exception as e:
        print(f"Error getting matches: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
