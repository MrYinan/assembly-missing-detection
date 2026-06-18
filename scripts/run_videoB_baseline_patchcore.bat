@echo off
setlocal
cd /d %~dp0\..
if not exist outputs_baseline mkdir outputs_baseline
if not exist figures_baseline mkdir figures_baseline
python -m src.pipeline --video-id B --video data\videoB_new.mp4 --config configs\videoB_deep_patchcore_baseline.json --normal-start 651 --normal-end 1011 --test-start 1011 --test-end -1 --train-fps 0.5 --infer-fps 1.0 --video-out-fps 5 --playback-speed 2.0 --model outputs_baseline\videoB_deep_patchcore_baseline_model.npz --output-dir outputs_baseline --retrain --render-video --disable-export-heatmaps
if errorlevel 1 goto fail
python -m src.generate_figures --frame-csv outputs_baseline\videoB_full_test_frame_predictions.csv --segments-json outputs_baseline\videoB_full_test_alarm_segments.json --out-dir figures_baseline --prefix videoB_baseline
if errorlevel 1 goto fail
python -m src.evaluate_keyframes --frame-csv outputs_baseline\videoB_full_test_frame_predictions.csv --labels labels\videoB_keyframes_full18min.csv --out outputs_baseline\videoB_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail
echo.
echo Done. Check outputs_baseline\videoB_full_test_side_by_side.mp4 and outputs_baseline\videoB_keyframe_eval.json
goto end
:fail
echo.
echo Failed. Check the error above.
:end
pause
