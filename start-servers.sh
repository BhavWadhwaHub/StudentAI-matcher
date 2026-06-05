#!/bin/bash

# ============================================================
#  Student AI Matcher - Start Servers (Mac / Linux)
#  Launches: Node.js API (port 3001) + Flask ML (port 5000)
# ============================================================

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get absolute project root from this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo " ============================================================"
echo "  Student AI Matcher - Server Launcher"
echo " ============================================================"
echo ""

# ── Node.js check ─────────────────────────────────────────────
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Node.js not found."
    echo "        Install from: https://nodejs.org/"
    exit 1
fi
NODE_VER=$(node --version 2>/dev/null)
echo -e "${GREEN}[OK]${NC} Node.js $NODE_VER found."

# ── Python check (prefer python3, fall back to python) ────────
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR]${NC} Python not found."
    echo "        Install from: https://www.python.org/"
    exit 1
fi
PY_VER=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}[OK]${NC} $PY_VER found."

# ── .env file check ───────────────────────────────────────────
echo ""
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${YELLOW}[WARN]${NC} .env file not found!"
    echo "       Copy .env.example to .env and fill in your credentials."
    echo "       Google Sheets integration and email sending will NOT work"
    echo "       until this is done."
    echo ""
else
    echo -e "${GREEN}[OK]${NC} .env file found."
fi

# ── Auto-install Node modules if missing ──────────────────────
if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo -e "${YELLOW}[SETUP]${NC} node_modules not found. Running npm install..."
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} npm install failed. Check your internet connection."
        exit 1
    fi
    echo -e "${GREEN}[OK]${NC} npm install complete."
    echo ""
fi

# ── Resolve Python for Flask (prefer venv over system Python) ─
FLASK_PY="$PYTHON_CMD"
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    FLASK_PY="$SCRIPT_DIR/venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    FLASK_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/env/bin/python" ]; then
    FLASK_PY="$SCRIPT_DIR/env/bin/python"
fi

if [ "$FLASK_PY" = "$PYTHON_CMD" ]; then
    echo -e "${CYAN}[INFO]${NC} No virtual environment found. Using system Python."
    echo "       To isolate ML dependencies, run once:"
    echo "         $PYTHON_CMD -m venv venv"
    echo "         source venv/bin/activate"
    echo "         pip install -r app_backend/requirements.txt"
    echo ""
else
    echo -e "${GREEN}[OK]${NC} Virtual environment found."
    echo "     Flask will use: $FLASK_PY"
    echo ""
fi

# ── Start Node.js server ──────────────────────────────────────
echo -e "${YELLOW}[1/3]${NC} Starting Node.js server  (http://localhost:3001)..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && npm start\"" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && npm start; exec bash" > /dev/null 2>&1
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "cd '$SCRIPT_DIR' && npm start; exec bash" > /dev/null 2>&1
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR' && npm start; exec bash" > /dev/null 2>&1
    else
        npm start &
        NODE_PID=$!
        echo "   Node.js server started in background (PID: $NODE_PID)"
    fi
else
    npm start &
    NODE_PID=$!
    echo "   Node.js server started in background (PID: $NODE_PID)"
fi

sleep 3

# ── Start Flask ML backend ────────────────────────────────────
echo -e "${YELLOW}[2/3]${NC} Starting Flask ML backend (http://localhost:5000)..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/app_backend' && '$FLASK_PY' app.py\"" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR/app_backend' && '$FLASK_PY' app.py; exec bash" > /dev/null 2>&1
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "cd '$SCRIPT_DIR/app_backend' && '$FLASK_PY' app.py; exec bash" > /dev/null 2>&1
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR/app_backend' && '$FLASK_PY' app.py; exec bash" > /dev/null 2>&1
    else
        cd "$SCRIPT_DIR/app_backend"
        "$FLASK_PY" app.py &
        FLASK_PID=$!
        cd "$SCRIPT_DIR"
        echo "   Flask server started in background (PID: $FLASK_PID)"
    fi
else
    cd "$SCRIPT_DIR/app_backend"
    "$FLASK_PY" app.py &
    FLASK_PID=$!
    cd "$SCRIPT_DIR"
    echo "   Flask server started in background (PID: $FLASK_PID)"
fi

echo ""
echo "Waiting for servers to initialize..."
sleep 5

# ── Open application in browser ───────────────────────────────
echo -e "${YELLOW}[3/3]${NC} Opening application in browser..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$SCRIPT_DIR/student-matcher.html" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "$SCRIPT_DIR/student-matcher.html" > /dev/null 2>&1
    elif command -v gnome-open &> /dev/null; then
        gnome-open "$SCRIPT_DIR/student-matcher.html" > /dev/null 2>&1
    else
        echo "   Please open this file in your browser manually:"
        echo "   $SCRIPT_DIR/student-matcher.html"
    fi
else
    if command -v xdg-open &> /dev/null; then
        xdg-open "$SCRIPT_DIR/student-matcher.html" > /dev/null 2>&1
    else
        echo "   Please open this file in your browser manually:"
        echo "   $SCRIPT_DIR/student-matcher.html"
    fi
fi

echo ""
echo -e "${GREEN} ============================================================${NC}"
echo -e "${GREEN} [OK] All servers launched!${NC}"
echo -e "${GREEN} ============================================================${NC}"
echo ""
echo "   Node.js  API  >  http://localhost:3001"
echo "   Flask ML API  >  http://localhost:5000"
echo "   Frontend      >  $SCRIPT_DIR/student-matcher.html"
echo ""
echo "   Servers are running in separate terminal windows."
echo "   Close those windows to stop the servers."
echo ""
