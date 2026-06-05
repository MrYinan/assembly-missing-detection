@echo off
setlocal
cd /d %~dp0\..
if not exist outputs mkdir outputs
if not exist figures mkdir figures
python -m src.pipeline --video-id A --video data\videoA.mp4 --config configs\videoA_deep_patchcore.json --normal-start 0 --normal-end 240 --test-start 240 --test-end -1 --train-fps 0.5 --infer-fps 1.0 --video-out-fps 5 --model outputs\videoA_deep_patchcore_model.npz --output-dir outputs --retrain --render-video --min-alarm-samples 8
if errorlevel 1 goto fail
python -m src.generate_figures --frame-csv outputs\videoA_full_test_frame_predictions.csv --segments-json outputs\videoA_full_test_alarm_segments.json --out-dir figures --prefix videoA
if errorlevel 1 goto fail
echo.
echo Done. Check outputs\videoA_full_test_side_by_side.mp4 and figures\videoA_full_score_curve.png
goto end
:fail
echo.
echo Failed. Check the error above. If the test segment is empty, verify data\videoA.mp4 duration and --test-start.
:end
pause
