@echo off
setlocal
cd /d %~dp0\..

echo.
echo ===== Evaluate Video A keyframes =====
python -m src.evaluate_keyframes ^
  --frame-csv outputs\videoA_full_test_frame_predictions.csv ^
  --labels labels\videoA_keyframes.csv ^
  --out outputs\videoA_keyframe_eval.json ^
  --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Evaluate Video B keyframes =====
python -m src.evaluate_keyframes ^
  --frame-csv outputs\videoB_full_test_frame_predictions.csv ^
  --labels labels\videoB_keyframes_full18min.csv ^
  --out outputs\videoB_keyframe_eval.json ^
  --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Build ablation table =====
python -m src.analyze_ablation ^
  --videos A B ^
  --output-dir outputs ^
  --config-dir configs ^
  --labels-dir labels ^
  --out-csv outputs\ablation_summary.csv ^
  --out-json outputs\ablation_summary.json ^
  --out-md outputs\ablation_summary.md ^
  --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo Done. Check outputs\videoA_keyframe_eval.json, outputs\videoB_keyframe_eval.json, and outputs\ablation_summary.md
goto end

:fail
echo.
echo Failed. Check the error above.

:end
pause
