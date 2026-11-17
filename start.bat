@echo off
setlocal
cd /d "%~dp0"
title AestheticEngine Launcher
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERREUR] Le script s'est termine avec une erreur.
    pause
)
