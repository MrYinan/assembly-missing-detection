@echo off
setlocal
cd /d %~dp0\..

if not exist figures_report mkdir figures_report

echo.
echo ===== Generate Video A report figures =====
python src\generate_report_figures_final.py ^
  --video-id A ^
  --video data\videoA.mp4 ^
  --normal-start 0 ^
  --normal-end 240 ^
  --test-start 240 ^
  --out-dir figures_report ^
  --clip-score 5

if errorlevel 1 goto fail

echo.
echo ===== Generate Video B report figures =====
python src\generate_report_figures_final.py ^
  --video-id B ^
  --video data\videoB_new.mp4 ^
  --normal-start 651 ^
  --normal-end 1011 ^
  --test-start 1011 ^
  --out-dir figures_report ^
  --clip-score 5

if errorlevel 1 goto fail

echo.
echo Done. Check figures_report folder.
goto end

:fail
echo.
echo Failed. Check the error above.

:end
pause