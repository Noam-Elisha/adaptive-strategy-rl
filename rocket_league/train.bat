@echo off
setlocal

cd /d "%~dp0"

:: --- Isolate embedded Python 3.13 from other installations ---
:: Do NOT set PYTHONHOME — auto-detection picks up Python 3.8
:: which is first on PATH and corrupts the embedded interpreter.
set PYTHONHOME=
set PYTHONNOUSERSITE=1
set PYTHONPATH=

:: Create python313._pth next to the exe to prevent the embedded
:: Python from loading any external site-packages (e.g. Python 3.8).
:: Lists only the correct Python 3.13 stdlib paths; skips site.py entirely.
if not exist "build\python313._pth" (
    (
        echo .
        echo C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\Lib
        echo C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\DLLs
    ) > "build\python313._pth"
    echo Created build\python313._pth for embedded Python isolation
)

:: --- Warn if exe is missing (dashboard will still load; build from UI) ---
if not exist "build\RocketLeagueStrategyBot.exe" (
    echo WARNING: build\RocketLeagueStrategyBot.exe not found.
    echo You can build from the dashboard before starting training.
    echo.
)

:: --- Open browser after a short delay ---
start "" "http://localhost:8050"

:: --- Launch monitor (starts dashboard, training controlled from browser) ---
:: Server prints its own banner + live action logs to this terminal
python monitor/server.py

pause
