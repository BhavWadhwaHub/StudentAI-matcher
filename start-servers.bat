@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Student AI Matcher - Start Servers (Windows)
REM  Launches: Node.js API (port 3001) + Flask ML (port 5000)
REM ============================================================

title Student AI Matcher

REM ── Resolve project root from this bat file's location ───────
set "ROOT=%~dp0"
if "!ROOT:~-1!"=="\" set "ROOT=!ROOT:~0,-1!"

echo.
echo  ============================================================
echo   Student AI Matcher - Server Launcher
echo  ============================================================
echo.

REM ── Node.js check ────────────────────────────────────────────
where node >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Node.js not found.
    echo         Install it from: https://nodejs.org/
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>nul') do set "NODE_VER=%%v"
echo [OK] Node.js !NODE_VER! found.

REM ── Python check (prefer "python" 3.11 over "py" launcher which may resolve to 3.13) ──
set "PYTHON_CMD="
where python >nul 2>&1
if !ERRORLEVEL! EQU 0 set "PYTHON_CMD=python"
if "!PYTHON_CMD!"=="" (
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 set "PYTHON_CMD=py"
)
if "!PYTHON_CMD!"=="" (
    echo [ERROR] Python not found.
    echo         Install it from: https://www.python.org/
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('!PYTHON_CMD! --version 2^>nul') do set "PY_VER=%%v"
echo [OK] !PY_VER! found.

REM ── .env file check ──────────────────────────────────────────
echo.
if not exist "!ROOT!\.env" (
    echo [WARN] .env file not found^^!
    echo        Copy .env.example to .env and fill in your credentials.
    echo        Google Sheets integration and email sending will NOT work
    echo        until this is done.
    echo.
) else (
    echo [OK] .env file found.
)

REM ── Auto-install Node modules if missing ─────────────────────
if not exist "!ROOT!\node_modules" (
    echo [SETUP] node_modules not found. Running npm install...
    pushd "!ROOT!"
    call npm install
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] npm install failed. Check your internet connection.
        pause & exit /b 1
    )
    popd
    echo [OK] npm install complete.
    echo.
)

REM ── Resolve Python for Flask (prefer venv over system Python) ─
set "FLASK_PY=!PYTHON_CMD!"
if exist "!ROOT!\venv\Scripts\python.exe"  set "FLASK_PY=!ROOT!\venv\Scripts\python.exe"
if exist "!ROOT!\.venv\Scripts\python.exe" set "FLASK_PY=!ROOT!\.venv\Scripts\python.exe"
if exist "!ROOT!\env\Scripts\python.exe"   set "FLASK_PY=!ROOT!\env\Scripts\python.exe"

if "!FLASK_PY!"=="!PYTHON_CMD!" (
    echo [INFO] No virtual environment found. Using system Python.
    echo        To isolate ML dependencies, run once:
    echo          !PYTHON_CMD! -m venv venv
    echo          venv\Scripts\activate
    echo          pip install -r app_backend\requirements.txt
    echo.
) else (
    echo [OK] Virtual environment found.
    echo      Flask will use: !FLASK_PY!
    echo.
)

REM ── Launch servers ────────────────────────────────────────────
echo [1/3] Starting Node.js server  ^(http://localhost:3001^)...
start "Node.js - Student Matcher" /d "!ROOT!" cmd /k "npm start"
timeout /t 3 /nobreak >nul

echo [2/3] Starting Flask ML backend ^(http://localhost:5000^)...
start "Flask ML - Student Matcher" /d "!ROOT!\app_backend" cmd /k ""!FLASK_PY!" app.py"
timeout /t 5 /nobreak >nul

echo [3/3] Opening application in browser...
start "" "!ROOT!\student-matcher.html"

echo.
echo  ============================================================
echo  [OK] All servers launched in separate windows^^!
echo  ============================================================
echo.
echo   Node.js  API  ^>  http://localhost:3001
echo   Flask ML API  ^>  http://localhost:5000
echo   Frontend      ^>  !ROOT!\student-matcher.html
echo.
echo   To stop: close the windows titled
echo     "Node.js - Student Matcher"  and  "Flask ML - Student Matcher"
echo.
pause >nul
endlocal
