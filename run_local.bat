@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "BACKEND_DIR=%ROOT%JINRO_PROJ\backend"
set "AI_DIR=%ROOT%JINRO_PROJ\ai_server"
set "FRONTEND_DIR=%ROOT%JINRO_PROJ\frontend"

if not exist "%VENV_PY%" (
  echo [ERROR] .venv Python not found: "%VENV_PY%"
  pause
  exit /b 1
)

if not exist "%BACKEND_DIR%\run.py" (
  echo [ERROR] Backend run.py not found: "%BACKEND_DIR%\run.py"
  pause
  exit /b 1
)

if not exist "%AI_DIR%\run.py" (
  echo [ERROR] AI server run.py not found: "%AI_DIR%\run.py"
  pause
  exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
  echo [ERROR] Frontend package.json not found: "%FRONTEND_DIR%\package.json"
  pause
  exit /b 1
)

echo Starting backend...
start "JINRO Backend" cmd /k "cd /d "%BACKEND_DIR%" && "%VENV_PY%" run.py"

echo Starting AI server...
start "JINRO AI Server" cmd /k "cd /d "%AI_DIR%" && "%VENV_PY%" run.py"

echo Starting frontend...
start "JINRO Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && npm.cmd run dev"

echo.
echo Local services are starting in separate windows.
echo Frontend: http://localhost:5173
echo Backend:  http://127.0.0.1:8000
echo AI:       http://127.0.0.1:8001
echo.
pause
