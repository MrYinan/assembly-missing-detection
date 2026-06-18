@echo off
setlocal
cd /d %~dp0\..
if not exist outputs mkdir outputs
if not exist figures mkdir figures
python -m src.pipeline --video-id B --video data\videoB_new.mp4 --config configs\videoB_deep_patchcore.json --normal-start 651 --normal-end 1011 --test-start 1011 --test-end -1 --train-fps 0.5 --infer-fps 1.0 --video-out-fps 5 --playback-speed 2.0 --model outputs\videoB_deep_patchcore_model.npz --output-dir outputs --retrain --render-video 
if errorlevel 1 goto fail
copy /Y outputs\videoB_full_test_side_by_side.mp4 outputs\videoB_full_test_side_by_side_full40s.mp4 >nul
if errorlevel 1 goto fail
python -m src.trim_video --input outputs\videoB_full_test_side_by_side_full40s.mp4 --output outputs\videoB_full_test_side_by_side_15s.mp4 --start-sec 19 --end-sec 34
if errorlevel 1 goto fail
python -m src.generate_figures --frame-csv outputs\videoB_full_test_frame_predictions.csv --segments-json outputs\videoB_full_test_alarm_segments.json --out-dir figures
if errorlevel 1 goto fail
python -m src.evaluate_keyframes --frame-csv outputs\videoB_full_test_frame_predictions.csv --labels labels\videoB_keyframes_full18min.csv --out outputs\videoB_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail
echo.
echo Done. Check outputs\videoB_full_test_side_by_side_15s.mp4, outputs\heatmaps\videoB, and outputs\videoB_keyframe_eval.json
goto end
:fail
echo.
echo Failed. If this is the first run, torchvision may be downloading ResNet weights. Check the error above.
:end
pause
