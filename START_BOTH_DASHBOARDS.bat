@echo off
echo ================================================
echo   Global LAB 1 - Starting Both Dashboards
echo ================================================
echo.

echo [1/2] Starting Air Quality Dashboard...
start "Air Quality Dashboard" cmd /k "cd 'Project 1 - Air Quality' && streamlit run app.py"
timeout /t 3 >nul

echo [2/2] Starting Tourism Dashboard...
start "Tourism Dashboard" cmd /k "cd 'Project 2 - Tourism' && streamlit run tourism_app.py"
timeout /t 3 >nul

echo.
echo ================================================
echo   Both Dashboards Starting!
echo ================================================
echo.
echo Project 1 (Air Quality): http://localhost:8501
echo Project 2 (Tourism):     http://localhost:8502
echo.
echo Press any key to exit...
pause >nul
