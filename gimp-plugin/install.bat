@echo off
REM Installs Moebius Inpaint for the current user's GIMP 3.2. Double-click it.
REM
REM   install.bat [weights-dir]
REM
REM Downloads about 2 GB of models from Hugging Face, checks them, and converts them.
REM Nothing else is needed. Run it again to resume or repair: finished models are skipped.
REM weights-dir, for developers, holds ft_places2\ and ft_celebahq\ already converted.
REM MOEBIUS_PLUGIN_DIR overrides the install folder.

setlocal EnableExtensions

set "SRC=%~dp0"
set "SELF=%~f0"
set "PROFILE=%APPDATA%\GIMP\3.2"
set "DEST=%MOEBIUS_PLUGIN_DIR%"
if not defined DEST set "DEST=%PROFILE%\plug-ins\moebius-inpaint"
set "WEIGHTS=%~1"

REM Pinned upstream revisions. moebius-convert checks every file against these hashes.
set "MOEBIUS_URL=https://huggingface.co/hustvl/Moebius/resolve/cd01f47fb648219d3fa605806c5ce00b713faa5e"
set "VAE_URL=https://huggingface.co/hustvl/PixelHacker/resolve/012fd343158936a265b8a0ee38a791a7a2841f45/vae"
set "VAE_SHA=a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe"
set "SCENES_SHA=6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a"
set "FACES_SHA=9d26630d91cf9748ed43541669197cf5d63fbb28793d0c1c5dc7c07028ad9045"

if not defined MOEBIUS_PLUGIN_DIR if not exist "%PROFILE%" (
    echo GIMP 3.2 was not found for this user.
    echo Install it from https://www.gimp.org/downloads/ and open it once, then run this again.
    goto :fail
)

REM A release has the programs next to this file. In a source checkout, build them.
set "BIN=%SRC%"
if exist "%BIN%moebius-cli.exe" if exist "%BIN%moebius-convert.exe" goto :have_programs
where cargo >nul 2>&1
if errorlevel 1 (
    echo moebius-cli.exe is missing. Download the release zip instead of the source code:
    echo https://github.com/CloudyTabzy/moebius-cpu/releases
    goto :fail
)
echo Building the programs, this takes a few minutes...
cargo build --release --features cli --bins --manifest-path "%SRC%..\Cargo.toml"
if errorlevel 1 (
    echo The build failed.
    goto :fail
)
set "BIN=%SRC%..\target\release\"

:have_programs
if not exist "%DEST%\models" mkdir "%DEST%\models"
if errorlevel 1 (
    echo Could not create %DEST%
    goto :fail
)
copy /Y "%SRC%moebius-inpaint.py" "%DEST%\" >nul || goto :copy_failed
copy /Y "%BIN%moebius-cli.exe" "%DEST%\" >nul || goto :copy_failed

if defined WEIGHTS goto :copy_models

where curl >nul 2>&1
if errorlevel 1 (
    echo curl was not found. It comes with Windows 10 version 1803 and newer.
    goto :fail
)
set "CACHE=%DEST%\download"
if not exist "%CACHE%\vae" mkdir "%CACHE%\vae"
call :model scenes ft_places2 %SCENES_SHA% || goto :fail
call :model faces ft_celebahq %FACES_SHA% || goto :fail
rmdir /S /Q "%CACHE%" 2>nul
goto :done

:copy_models
for %%P in ("ft_places2 scenes" "ft_celebahq faces") do for /F "tokens=1,2" %%A in (%%P) do (
    if not exist "%DEST%\models\%%B" mkdir "%DEST%\models\%%B"
    for %%F in (unet vae) do (
        if not exist "%WEIGHTS%\%%A\%%F.safetensors" (
            echo Missing %WEIGHTS%\%%A\%%F.safetensors
            goto :fail
        )
        copy /Y "%WEIGHTS%\%%A\%%F.safetensors" "%DEST%\models\%%B\" >nul || goto :copy_failed
    )
)

:done
echo.
echo Installed to %DEST%
echo Restart GIMP, select what to remove, then use Filters ^> Enhance ^> Moebius Inpaint...
call :pause_if_clicked
exit /b 0

:copy_failed
echo Copying to %DEST% failed. If GIMP is running, close it and try again.

:fail
echo.
echo The installation did not finish.
call :pause_if_clicked
exit /b 1

REM :model <folder> <upstream name> <sha256>
:model
set "OUT=%DEST%\models\%~1"
if exist "%OUT%\unet.safetensors" if exist "%OUT%\vae.safetensors" (
    echo The %~1 model is already installed.
    exit /b 0
)
echo.
echo Downloading the %~1 model...
call :fetch "%MOEBIUS_URL%/%~2/diffusion_pytorch_model.bin" "%CACHE%\%~2.bin" || exit /b 1
call :fetch "%VAE_URL%/config.json" "%CACHE%\vae\config.json" || exit /b 1
call :fetch "%VAE_URL%/diffusion_pytorch_model.bin" "%CACHE%\vae\diffusion_pytorch_model.bin" || exit /b 1
echo Converting the %~1 model...
"%BIN%moebius-convert.exe" --unet "%CACHE%\%~2.bin" --unet-sha256 %~3 --vae "%CACHE%\vae" --vae-sha256 %VAE_SHA% --out "%OUT%"
if not "%ERRORLEVEL%"=="0" (
    echo The downloaded files failed their check, so they were deleted. Run this again.
    del /Q "%CACHE%\%~2.bin" "%CACHE%\vae\diffusion_pytorch_model.bin" 2>nul
    exit /b 1
)
del /Q "%CACHE%\%~2.bin" 2>nul
exit /b 0

REM :fetch <url> <file>. Large downloads often drop, so it keeps resuming the .part file.
:fetch
if exist "%~2" exit /b 0
set /a TRIES=0
:fetch_retry
curl -fL -# --speed-limit 1024 --speed-time 60 -C - -o "%~2.part" "%~1"
if "%ERRORLEVEL%"=="0" goto :fetched
set /a TRIES+=1
if %TRIES% GEQ 30 (
    echo The download failed. Check your connection and that about 3 GB of disk is free, then run this again.
    exit /b 1
)
echo The connection dropped. Resuming in 5 seconds, attempt %TRIES% of 30...
ping -n 6 127.0.0.1 >nul
goto :fetch_retry
:fetched
move /Y "%~2.part" "%~2" >nul
exit /b 0

REM Keeps the window open when the script was double-clicked.
:pause_if_clicked
echo %CMDCMDLINE% | "%SystemRoot%\System32ind.exe" /I "%SELF%" >nul
if not errorlevel 1 pause
exit /b 0
