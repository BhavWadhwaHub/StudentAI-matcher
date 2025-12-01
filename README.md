# 🎯 Neura Linker AI - Student Matcher

An intelligent AI-powered student matching system that connects students with compatible study partners based on academic interests, personality traits, and shared hobbies. The system uses advanced machine learning algorithms including K-means clustering, weighted similarity matching, and Jaccard Index calculations to find the best matches.

**Authors:** Bhav Wadhwa, Mohammed Mujtaba Fatah, Madina Alizada, Anelya Shaimardanova

---

## 📋 Table of Contents
- [Features](#-features)
- [Technical Architecture](#-technical-architecture)
- [Matching Algorithm Logic](#-matching-algorithm-logic)
- [Technologies Used](#-technologies-used)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Evaluation & Testing](#-evaluation--testing)
- [API Documentation](#-api-documentation)
- [Quick Installation Guide](#-quick-installation-guide)

---

## ✨ Features

### Core Functionality
- 🤖 **AI-Powered Matching** - Advanced algorithms analyze student profiles to find compatible study partners
- 📊 **Multiple Matching Strategies**:
  - Weighted Similarity Matching (pairwise comparison)
  - K-means Clustering (group formation)
  - Hybrid Approach (combines both methods)
- 🎯 **Smart Scoring** - Comprehensive compatibility scores based on 9 different attributes
- 🌈 **Diversity Enforcement** - Promotes diverse matches to prevent cultural clustering
- 📧 **Automated Email Notifications** - Beautiful HTML emails with match details and compatibility scores
- 💾 **Google Sheets Integration** - Real-time data storage and synchronization
- 🎨 **Modern UI** - Beautiful, responsive interface with dark/light mode support
- 📱 **Mobile Responsive** - Optimized for phone, tablet, and desktop views

### Advanced Features
- **Dynamic Threshold Adjustment** - Adapts minimum similarity scores based on dataset size
- **Tie-Breaking Logic** - Intelligent handling of equal-score matches
- **Performance Optimization** - Efficient algorithms handle large student populations (600+ students)
- **Comprehensive Evaluation** - Built-in metrics for accuracy, coverage, diversity, and stability
- **Configurable Weights** - Easy-to-adjust matching parameters

---

## 🏗 Technical Architecture

The system consists of three main components:

### 1. **Frontend (HTML/CSS/JavaScript)**
   - `student-matcher.html` - Main user interface
   - `styles.css` - Styling with dark/light mode support
   - `script.js` - Client-side form handling and API communication

### 2. **Node.js Backend (Express Server)**
   - `server.js` - Handles HTTP requests, Google Sheets integration, and email delivery
   - Port: `3001`
   - Responsibilities:
     - Serve frontend files
     - Store student data in Google Sheets
     - Send match notification emails
     - Communicate with Flask backend for AI matching

### 3. **Python Flask Backend (AI Engine)**
   - `app_backend/app.py` - Flask server with matching endpoints
   - `app_backend/matcher.py` - Core matching algorithms
   - `app_backend/config.py` - Configuration and weights
   - Port: `5000`
   - Responsibilities:
     - Load student data from Google Sheets API
     - Execute matching algorithms
     - Return computed matches to Node.js server

### Data Flow
```
User → Frontend → Node.js (port 3001) → Google Sheets
                       ↓
                Flask (port 5000) → Matching Algorithms
                       ↓
                Email Service (Gmail SMTP)
                       ↓
                User receives matches
```

---

## 🧠 Matching Algorithm Logic

### Algorithm Options

#### 1. **Weighted Similarity Matching**
Calculates pairwise similarity between students using weighted attributes:

**Weights:**
- Academic Major: 30%
- Personality Type: 25%
- Interests (Jaccard Index): 22%
- Study Style: 7%
- Year of Study: 6%
- Country: 5% (reduced to promote diversity)
- Cuisine: 2%
- Movies: 2%
- Language: 1% (minimized to avoid clustering)

**Process:**
1. For each student, compare with all other students
2. Calculate weighted similarity score (0-100%)
3. Apply Jaccard Index for set-based attributes (interests, cuisine, movies)
4. Normalize scores using min-max scaling
5. Enforce diversity bonuses (up to +5% for diverse matches)
6. Select top N matches above minimum threshold

#### 2. **K-means Clustering**
Groups students into clusters based on feature similarity:

**Process:**
1. Encode categorical features (one-hot encoding)
2. Vectorize set-based features (multi-label binarization)
3. Combine all features into a unified feature matrix
4. Apply K-means clustering algorithm
5. Match students within the same cluster
6. Calculate similarity scores within clusters

**Parameters:**
- Number of clusters: Determined automatically using Elbow Method (tests k=2 to k=12)
- Maximum clusters: 20 (configurable)
- Minimum cluster size: 2 students

#### 3. **Hybrid Matching**
Combines weighted similarity and clustering for optimal results:

**Process:**
1. Run K-means clustering to group similar students
2. Within each cluster, use weighted similarity matching
3. Blend results using configurable weights (70% similarity, 30% clustering)
4. This approach balances precision and computational efficiency

### Diversity Enforcement

To prevent cultural clustering and promote diverse connections:

**Diversity Attributes:**
- Country of origin
- Language spoken
- Academic major
- Year of study

**Diversity Bonuses:**
- Country difference: +3% bonus
- Major difference: +2% bonus
- Language difference: +1% bonus
- Year difference: +0.5% bonus

**Maximum diversity bonus:** 5% (capped to maintain score integrity)

### Similarity Calculation Methods

#### Exact Match (Categorical)
```python
similarity = 1.0 if value1 == value2 else 0.0
```

#### Jaccard Index (Sets)
```python
jaccard = len(set1 ∩ set2) / len(set1 ∪ set2)
```
Used for: interests, cuisine preferences, movie genres

#### Normalized Difference (Ordinal)
```python
similarity = 1 - |value1 - value2| / max_range
```
Used for: year of study

---

## 💻 Technologies Used

### Backend (Python)
- **Flask** 3.0.2 - Web framework
- **Flask-CORS** 6.0.1 - Cross-origin resource sharing
- **Pandas** 2.2.3 - Data manipulation
- **NumPy** 1.26.4 - Numerical computing
- **scikit-learn** 1.5.2 - Machine learning (K-means)
- **Requests** 2.32.3 - HTTP client
- **Matplotlib** 3.8.2 - Visualization (for evaluation)

### Backend (Node.js)
- **Express** 5.1.0 - Web server
- **CORS** 2.8.5 - Cross-origin support
- **googleapis** 161.0.0 - Google Sheets API integration
- **nodemailer** 7.0.10 - Email sending
- **dotenv** 17.2.3 - Environment configuration
- **node-fetch** 3.3.2 - HTTP client

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with modern features
  - CSS Grid & Flexbox
  - CSS Custom Properties
  - Media Queries (responsive design)
  - Dark/Light mode support
- **Vanilla JavaScript** - Client-side logic
- **Google Fonts** (Inter) - Typography

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v14 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://www.python.org/)
- **npm** (comes with Node.js)
- **pip** (comes with Python)
- **Google Cloud Account** - For Google Sheets API
- **Gmail Account** - For email notifications

---

## 🛠 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd studentmatcher
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Install Python Dependencies
```bash
cd app_backend
pip install -r requirements.txt
cd ..
```

### 4. Set Up Google Sheets API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable Google Sheets API
4. Create a Service Account:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Grant it "Editor" role
   - Create a JSON key and download it
5. Create a Google Sheet and note its ID (from URL)
6. Share the sheet with the service account email (from JSON key)

### 5. Set Up Gmail for Email Notifications

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Step Verification
3. Go to [App Passwords](https://myaccount.google.com/apppasswords)
4. Generate a new App Password for "Mail"
5. Copy the 16-character password

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Google Sheets Configuration
SPREADSHEET_ID=your_spreadsheet_id_here
GOOGLE_CREDENTIALS={"type":"service_account","project_id":"...","private_key":"..."}

# Email Configuration
EMAIL_USER=your_gmail_address@gmail.com
EMAIL_PASSWORD=your_16_character_app_password

# Server Configuration
PORT=3001
```

**Important Notes:**
- `GOOGLE_CREDENTIALS` should be the entire JSON content from your service account key (as a single line)
- `EMAIL_PASSWORD` must be exactly 16 characters (Gmail App Password)
- Never commit `.env` to version control

---

## 🚀 How to Run

### Option 1: Using Platform-Specific Scripts (Recommended)

#### Windows:
```bash
start-servers.bat
```

#### Mac/Linux:
```bash
chmod +x start-servers.sh
./start-servers.sh
```

These scripts will:
1. Check if Node.js and Python are installed
2. Start the Node.js server (port 3001)
3. Start the Flask backend (port 5000)
4. Wait 5 seconds for initialization
5. Automatically open `student-matcher.html` in your browser

### Option 2: Manual Start

#### Terminal 1 - Node.js Server:
```bash
npm start
```
Server will start on `http://localhost:3001`

#### Terminal 2 - Flask Backend:
```bash
cd app_backend
python app.py
```
Server will start on `http://localhost:5000`

#### Terminal 3 - Open Frontend:
```bash
# Windows
start student-matcher.html

# Mac
open student-matcher.html

# Linux
xdg-open student-matcher.html
```

### Verify Everything is Running

You should see:
- ✅ Node.js server running on port 3001
- ✅ Flask server running on port 5000
- ✅ Google Sheets connection successful
- ✅ Frontend loaded in browser

---

## 📁 Project Structure

```
studentmatcher/
│
├── app_backend/                # Python Flask backend
│   ├── app.py                  # Main Flask application
│   ├── matcher.py              # Core matching algorithms
│   ├── config.py               # Configuration and weights
│   ├── requirements.txt        # Python dependencies
│   └── venv/                   # Python virtual environment
│
├── evaluation/                 # Evaluation and metrics
│   └── comprehensive_evaluation.py  # Model evaluation suite
│
├── node_modules/               # Node.js dependencies (auto-generated)
│
├── student-matcher.html        # Main frontend interface
├── styles.css                  # Styling (with dark mode)
├── script.js                   # Frontend JavaScript
├── server.js                   # Node.js Express server
│
├── package.json                # Node.js dependencies
├── package-lock.json           # Locked dependency versions
│
├── start-servers.bat           # Windows startup script
├── start-servers.sh            # Mac/Linux startup script
│
├── .env                        # Environment configuration (create this)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── INSTALLATION.md             # Quick installation guide
```

---

## ⚙️ Configuration

### Matching Weights (`app_backend/config.py`)

Adjust attribute importance:

```python
MATCHING_WEIGHTS = {
    'major': 30,          # Academic major
    'personality': 25,    # Personality type
    'interests': 22,      # Hobbies/interests
    'country': 5,         # Country (low to promote diversity)
    'year': 6,            # Academic year
    'studyStyle': 7,      # Study style preference
    'language': 1,        # Language (low to promote diversity)
    'cuisine': 2,         # Food preferences
    'movies': 2           # Movie preferences
}
```

### Matching Parameters

```python
# Number of top matches to return
TOP_MATCHES_COUNT = 3

# Minimum similarity score (0-100) - Dynamic based on dataset size
# 2-5 students: 25%, 6-10 students: 35%, 11+ students: 38%
MIN_SIMILARITY_SCORE = 25  # Base minimum, adjusted dynamically

# Algorithm selection
MATCHING_ALGORITHM = 'kmeans'  # or 'weighted_similarity', 'hybrid' (default: 'kmeans')

# Diversity enforcement
ENABLE_DIVERSITY_ENFORCEMENT = True
MAX_DIVERSITY_BONUS = 5
```

### Similarity Thresholds

```python
SIMILARITY_THRESHOLDS = {
    'excellent': 80,      # 80-100%
    'very_good': 60,      # 60-79%
    'good': 40,           # 40-59%
    'decent': 0           # 0-39%
}
```

---

## 📊 Evaluation & Testing

### Running Comprehensive Evaluation

```bash
cd evaluation
python comprehensive_evaluation.py
```

This generates:

**Metrics:**
- ✅ Basic Metrics: match counts, coverage, score distribution
- ✅ Quality Metrics: high-quality match rates, average scores
- ✅ Diversity Metrics: country diversity, language diversity, major diversity
- ✅ Clustering Metrics: silhouette score, cluster distribution
- ✅ Stability Metrics: consistency across multiple runs

**Outputs:**
- `evaluation_results/` folder with:
  - Detailed metrics report (JSON and text)
  - Visualization plots (PNG images)
  - Statistical analysis

### Performance Benchmarks

**Optimized for large datasets:**
- 600+ students: ~15-30 seconds (optimized from 5+ minutes)
- Uses pre-computed dictionaries and indexed iteration
- Avoids nested DataFrame operations

---

## 🔌 API Documentation

### Node.js Server (Port 3001)

#### GET `/api/students`
Fetches all students from Google Sheets.

**Response:**
```json
[
  {
    "name": "John Doe",
    "email": "john@university.edu",
    "major": "Computer Science",
    "year": "Sophomore",
    "language": "English",
    "country": "USA",
    "personality": "Extrovert",
    "studyStyle": "Group Study",
    "cuisine": ["Italian", "Chinese"],
    "interests": ["Coding", "Gaming"],
    "movies": ["Sci-Fi", "Action"],
    "timestamp": "2025-01-01T12:00:00Z"
  }
]
```

#### POST `/api/students`
Adds a new student and triggers matching.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@university.edu",
  "major": "Computer Science",
  "year": "Sophomore",
  "language": "English",
  "country": "USA",
  "personality": "Introvert",
  "studyStyle": "Solo Study",
  "cuisine": ["Mexican", "Thai"],
  "interests": ["Reading", "Art"],
  "movies": ["Drama", "Comedy"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Student added successfully",
  "student": { /* student data */ },
  "matches": [
    {
      "name": "John Doe",
      "email": "john@university.edu",
      "similarity_score": 85.5,
      "match_quality": "Excellent",
      "commonalities": [
        "Same major: Computer Science",
        "Same year: Sophomore",
        "Shared interest: Coding"
      ]
    }
  ],
  "totalMatches": 1,
  "emailSent": true
}
```

### Flask Backend (Port 5000)

#### GET `/api/match?name=<student_name>`
Computes matches for a specific student.

**Response:**
```json
{
  "success": true,
  "student_name": "Jane Smith",
  "matches": [ /* array of matches */ ],
  "total_matches": 3,
  "algorithm_used": "weighted_similarity"
}
```

#### GET `/api/match-all`
Computes matches for all students.

**Response:**
```json
{
  "success": true,
  "total_students": 100,
  "all_matches": {
    "Jane Smith": [ /* matches */ ],
    "John Doe": [ /* matches */ ]
  }
}
```

---

## 🎨 UI Features

### Dark/Light Mode Support
The interface automatically adapts to the user's system preference using CSS media queries:
```css
@media (prefers-color-scheme: dark) {
  /* Dark mode styles */
}
```

### Responsive Design
Optimized breakpoints:
- **Desktop:** > 768px (full layout)
- **Tablet:** 480px - 768px (adjusted layout)
- **Mobile:** < 480px (single column)

### Screens
1. **Welcome Screen** - Hero section with features
2. **Form Screen** - Student profile input
3. **Processing Screen** - Loading animation
4. **Results Screen** - Success message with match details

---

## 📖 Quick Installation Guide

For a quick start, see [INSTALLATION.md](INSTALLATION.md) for simplified installation steps.

---

## 🐛 Troubleshooting

### Common Issues

**1. Flask backend not responding**
```bash
# Check if Flask is running
curl http://localhost:5000/api/match-all

# Restart Flask
cd app_backend
python app.py
```

**2. Google Sheets connection failed**
- Verify `SPREADSHEET_ID` in `.env`
- Check service account has edit access to sheet
- Ensure `GOOGLE_CREDENTIALS` is valid JSON

**3. Email not sending**
- Verify `EMAIL_USER` and `EMAIL_PASSWORD` in `.env`
- Ensure password is 16-character App Password
- Check 2-Step Verification is enabled

**4. ModuleNotFoundError (Python)**
```bash
cd app_backend
pip install -r requirements.txt
```

**5. Port already in use**
```bash
# Kill process on port 3001 (Node.js)
# Windows:
netstat -ano | findstr :3001
taskkill /PID <PID> /F

# Mac/Linux:
lsof -ti:3001 | xargs kill -9
```

---

## 📝 License

ISC License

---

## 👥 Authors

- **Bhav Wadhwa**
- **Mohammed Mujtaba Fatah**
- **Madina Alizada**
- **Anelya Shaimardanova**

---

## 🙏 Acknowledgments

- Powered by scikit-learn for machine learning
- Google Sheets API for data storage
- Express.js and Flask for backend services
- Inter font family by Rasmus Andersson

---


---

**Made with ❤️ for connecting students worldwide**

