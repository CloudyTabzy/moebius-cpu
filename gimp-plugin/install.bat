@echo off
REM Double-click to install Moebius Inpaint. This only finds a Python and runs install.py,
REM preferring the one that comes with GIMP, so nothing else needs installing.

setlocal EnableExtensions
set "SELF=%~f0"
set "PY="
if exist "%LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe" set "PY=%LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe"
if not defined PY if exist "%ProgramFiles%\GIMP 3\bin\python.exe" set "PY=%ProgramFiles%\GIMP 3\bin\python.exe"
if defined PY goto :run

REM GIMP is somewhere else: any working Python will do. Windows' Store stub fails this test.
python -c "import ssl" >nul 2>&1
if "%ERRORLEVEL%"=="0" set "PY=python"
if defined PY goto :run
echo GIMP 3.2 was not found.
echo Install it from https://www.gimp.org/downloads/ and open it once, then run this again.
set "CODE=1"
goto :end

:run
"%PY%" -I -u "%~dp0install.py" %*
set "CODE=%ERRORLEVEL%"

:end
REM Keeps the window open when the script was double-clicked.
echo %CMDCMDLINE% | "%SystemRoot%\System32\find.exe" /I "%SELF%" >nul
if "%ERRORLEVEL%"=="0" pause
exit /b %CODE%
