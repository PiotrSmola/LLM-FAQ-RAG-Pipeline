@echo off
rem Uruchomienie calego projektu w Dockerze (szczegoly i polecenia: start.ps1, README.md)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
