@echo off
REM Installs Moebius Inpaint for the current user's GIMP 3.2.
REM
REM   install.bat [weights-dir]
REM
REM weights-dir holds ft_places2\ and ft_celebahq\ from tools\convert_weights.py
REM (default: ..\weights). Uses a prebuilt moebius-cli.exe next to this file, or
REM builds one with cargo. Needs about 1.3 GB of disk.

setlocal EnableExtensions

set "SRC=%~dp0"
set "DEST=%APPDATA%\GIMP\3.2\plug-ins\moebius-inpaint"
set "WEIGHTS=%~1"
if not defined WEIGHTS set "WEIGHTS=%SRC%..\weights"

for %%M in (ft_places2 ft_celebahq) do (
    for %%F in (unet vae) do (
        if not exist "%WEIGHTS%\%%M\%%F.safetensors" (
            echo ERROR: Missing %WEIGHTS%\%%M\%%F.safetensors
            echo Convert the checkpoints first: see docs\PLAN.md section 8.
            exit /b 1
        )
    )
)

set "WORKER=%SRC%moebius-cli.exe"
if exist "%WORKER%" goto :have_worker
where cargo >nul 2>&1
if errorlevel 1 (
    echo ERROR: moebius-cli.exe is not here and cargo is not on PATH.
    exit /b 1
)
echo Building moebius-cli, this takes a few minutes...
cargo build --release --features cli --bin moebius-cli --manifest-path "%SRC%..\Cargo.toml"
if errorlevel 1 (
    echo ERROR: The build failed.
    exit /b 1
)
set "WORKER=%SRC%..\target\release\moebius-cli.exe"

:have_worker
if not exist "%DEST%\models\scenes" mkdir "%DEST%\models\scenes"
if not exist "%DEST%\models\faces" mkdir "%DEST%\models\faces"
if errorlevel 1 (
    echo ERROR: Could not create %DEST%
    exit /b 1
)

copy /Y "%SRC%moebius-inpaint.py" "%DEST%\" >nul || goto :copy_failed
copy /Y "%WORKER%" "%DEST%\moebius-cli.exe" >nul || goto :copy_failed
echo Copying models...
for %%F in (unet vae) do (
    copy /Y "%WEIGHTS%\ft_places2\%%F.safetensors" "%DEST%\models\scenes\" >nul || goto :copy_failed
    copy /Y "%WEIGHTS%\ft_celebahq\%%F.safetensors" "%DEST%\models\faces\" >nul || goto :copy_failed
)

echo.
echo Installed to %DEST%
echo Restart GIMP, select what to remove, then use Filters ^> Enhance ^> Moebius Inpaint...
exit /b 0

:copy_failed
echo ERROR: Copying to %DEST% failed. If GIMP is running, close it and try again.
exit /b 1
