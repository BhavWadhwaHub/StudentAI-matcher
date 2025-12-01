@echo off
REM Student Matcher - Start Servers Script for Windows
REM This script starts both Flask and Node.js servers

echo ========================================
echo  Student Matcher - Starting Servers
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/2] Starting Node.js Server (Port 3001)...
start "Node.js Server" cmd /k "npm start"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Flask Backend Server (Port 5000)...
cd app_backend
start "Flask Server" cmd /k "python app.py"
cd ..

echo.
echo Waiting for servers to initialize...
timeout /t 5 /nobreak >nul

echo [3/3] Opening application in browser...
start "" "student-matcher.html"

echo.
echo ========================================
echo [OK] Both servers are starting!
echo ========================================
echo.
echo Node.js Server: http://localhost:3001
echo Flask Backend:  http://localhost:5000
echo Application:    student-matcher.html
echo.
echo Press any key to close this window (servers will continue running)...
pause >nul

