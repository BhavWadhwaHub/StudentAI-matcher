#!/bin/bash

# Student Matcher - Start Servers Script for Mac/Linux
# This script starts both Flask and Node.js servers

echo "========================================"
echo " Student Matcher - Starting Servers"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python is not installed or not in PATH"
    echo "Please install Python from https://www.python.org/"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}[1/2]${NC} Starting Node.js Server (Port 3001)..."
# Start Node.js server in a new terminal window (macOS/Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && npm start\"" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && npm start; exec bash" > /dev/null 2>&1
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR' && npm start" > /dev/null 2>&1
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "cd '$SCRIPT_DIR' && npm start; exec bash" > /dev/null 2>&1
    else
        # Fallback: run in background
        npm start &
        NODE_PID=$!
        echo "   Node.js server started in background (PID: $NODE_PID)"
    fi
else
    # Fallback: run in background
    npm start &
    NODE_PID=$!
    echo "   Node.js server started in background (PID: $NODE_PID)"
fi

sleep 3

echo -e "${YELLOW}[2/2]${NC} Starting Flask Backend Server (Port 5000)..."
# Start Flask server in a new terminal window (macOS/Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR/app_backend' && $PYTHON_CMD app.py\"" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$SCRIPT_DIR/app_backend' && $PYTHON_CMD app.py; exec bash" > /dev/null 2>&1
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$SCRIPT_DIR/app_backend' && $PYTHON_CMD app.py" > /dev/null 2>&1
    elif command -v konsole &> /dev/null; then
        konsole -e bash -c "cd '$SCRIPT_DIR/app_backend' && $PYTHON_CMD app.py; exec bash" > /dev/null 2>&1
    else
        # Fallback: run in background
        cd app_backend
        $PYTHON_CMD app.py &
        FLASK_PID=$!
        cd ..
        echo "   Flask server started in background (PID: $FLASK_PID)"
    fi
else
    # Fallback: run in background
    cd app_backend
    $PYTHON_CMD app.py &
    FLASK_PID=$!
    cd ..
    echo "   Flask server started in background (PID: $FLASK_PID)"
fi

echo ""
echo "Waiting for servers to initialize..."
sleep 5

echo -e "${YELLOW}[3/3]${NC} Opening application in browser..."
# Open HTML file in default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "student-matcher.html" > /dev/null 2>&1
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open "student-matcher.html" > /dev/null 2>&1
    elif command -v gnome-open &> /dev/null; then
        gnome-open "student-matcher.html" > /dev/null 2>&1
else
        echo "   Please open student-matcher.html in your browser manually"
    fi
else
    # Fallback
    if command -v xdg-open &> /dev/null; then
        xdg-open "student-matcher.html" > /dev/null 2>&1
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}[OK] Both servers are starting!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Node.js Server: http://localhost:3001"
echo "Flask Backend:  http://localhost:5000"
echo "Application:    student-matcher.html"
echo ""
echo "Servers are running in separate terminal windows."
echo "Close those windows to stop the servers."

