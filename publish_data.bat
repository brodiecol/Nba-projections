@echo off
cd /d "%~dp0"
echo Pulling today's NBA data...
py refresh_data.py
if errorlevel 1 (
  echo.
  echo The data pull failed, so nothing was uploaded.
  pause
  exit /b 1
)
echo.
echo Uploading data\latest.json to GitHub...
git add data/latest.json
git commit -m "Update NBA data"
git push
echo.
echo Done. The online app updates within a minute or two.
pause
