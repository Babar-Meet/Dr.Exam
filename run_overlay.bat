@echo off
setlocal
cd /d %~dp0
python -m proctor_app.main --view-mode overlay
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
	echo.
	echo run_overlay failed with exit code %exitCode%.
	pause
)
exit /b %exitCode%
