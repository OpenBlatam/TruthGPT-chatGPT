@echo off
REM Test runner for TruthGPT optimization core

echo ========================================
echo TruthGPT Optimization Core Test Suite
echo ========================================
echo.

REM Try different Python commands
where py 2>nul >nul
if %ERRORLEVEL% == 0 (
    echo Using Python Launcher...
    py tests\run_all_tests.py %*
    goto :end
)

where python3 2>nul >nul
if %ERRORLEVEL% == 0 (
    echo Using python3...
    python3 tests\run_all_tests.py %*
    goto :end
)

where python 2>nul >nul
if %ERRORLEVEL% == 0 (
    echo Using python...
    python tests\run_all_tests.py %*
    goto :end
)

echo Python not found! Please install Python first.
echo.
echo Installing Python from Microsoft Store or python.org
pause
exit /b 1

:end
echo.
echo Tests completed!
pause


