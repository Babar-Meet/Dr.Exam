@echo off
setlocal
cd /d %~dp0

echo Opening test exam page in browser...
start "" "darshanums.in\test_exam.html"

echo Waiting 3 seconds for browser to open...
timeout /t 3 /nobreak >nul

echo Starting proctor monitor...
python -m proctor_app.main --view-mode overlay

endlocal