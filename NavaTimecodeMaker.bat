@echo off
REM Builds NavaTimecodeMaker.exe. Run from the folder containing tc_app_v3.py.
REM Uses "python -m" so it works even when pyinstaller is not on PATH.

python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

python -m PyInstaller --onefile --clean --name NTM tc_app_v3.py
if errorlevel 1 goto :fail

echo.
echo Build finished: dist\NTM3.2.exe
pause
exit /b 0

:fail
echo.
echo Build failed. See the messages above.
pause
exit /b 1
