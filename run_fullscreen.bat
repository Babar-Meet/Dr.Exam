@echo off
setlocal
cd /d %~dp0
python -m proctor_app.main --view-mode fullscreen
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
	echo.
	echo run_fullscreen failed with exit code %exitCode%.
	pause
)
exit /b %exitCode%
