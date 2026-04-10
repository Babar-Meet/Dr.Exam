@echo off
setlocal
cd /d %~dp0
python -m proctor_app.state_logger_main --interval-seconds 1.0
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
	echo.
	echo run_state_logger failed with exit code %exitCode%.
	pause
)
exit /b %exitCode%
