@echo off
setlocal
cd /d %~dp0\..
if not exist outputs_full_eval mkdir outputs_full_eval
if not exist outputs_baseline_eval mkdir outputs_baseline_eval
if not exist outputs_eval_compare mkdir outputs_eval_compare

echo.
echo ===== Enhanced full evaluation: Video A =====
python -B -m src.pipeline --video-id A --video data\videoA.mp4 --config configs\videoA_deep_patchcore_full_eval.json --normal-start 0 --normal-end 240 --test-start 240 --test-end 9999 --train-fps 0.5 --infer-fps 1.0 --model outputs_full_eval\videoA_deep_patchcore_model.npz --output-dir outputs_full_eval --retrain
if errorlevel 1 goto fail
python -B -m src.evaluate_keyframes --frame-csv outputs_full_eval\videoA_full_test_frame_predictions.csv --labels labels\videoA_keyframes.csv --out outputs_full_eval\videoA_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Enhanced full evaluation: Video B =====
python -B -m src.pipeline --video-id B --video data\videoB_new.mp4 --config configs\videoB_deep_patchcore.json --normal-start 651 --normal-end 1011 --test-start 1011 --test-end -1 --train-fps 0.5 --infer-fps 1.0 --model outputs_full_eval\videoB_deep_patchcore_model.npz --output-dir outputs_full_eval --retrain
if errorlevel 1 goto fail
python -B -m src.evaluate_keyframes --frame-csv outputs_full_eval\videoB_full_test_frame_predictions.csv --labels labels\videoB_keyframes_full18min.csv --out outputs_full_eval\videoB_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Baseline full evaluation: Video A =====
python -B -m src.pipeline --video-id A --video data\videoA.mp4 --config configs\videoA_deep_patchcore_baseline_full_eval.json --normal-start 0 --normal-end 240 --test-start 240 --test-end 9999 --train-fps 0.5 --infer-fps 1.0 --model outputs_baseline_eval\videoA_deep_patchcore_baseline_model.npz --output-dir outputs_baseline_eval --retrain --disable-export-heatmaps
if errorlevel 1 goto fail
python -B -m src.evaluate_keyframes --frame-csv outputs_baseline_eval\videoA_full_test_frame_predictions.csv --labels labels\videoA_keyframes.csv --out outputs_baseline_eval\videoA_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Baseline full evaluation: Video B =====
python -B -m src.pipeline --video-id B --video data\videoB_new.mp4 --config configs\videoB_deep_patchcore_baseline.json --normal-start 651 --normal-end 1011 --test-start 1011 --test-end -1 --train-fps 0.5 --infer-fps 1.0 --model outputs_baseline_eval\videoB_deep_patchcore_baseline_model.npz --output-dir outputs_baseline_eval --retrain --disable-export-heatmaps
if errorlevel 1 goto fail
python -B -m src.evaluate_keyframes --frame-csv outputs_baseline_eval\videoB_full_test_frame_predictions.csv --labels labels\videoB_keyframes_full18min.csv --out outputs_baseline_eval\videoB_keyframe_eval.json --tolerance 1.0
if errorlevel 1 goto fail

echo.
echo ===== Compare enhanced vs baseline =====
python -B -m src.compare_eval_runs --enhanced-dir outputs_full_eval --baseline-dir outputs_baseline_eval --videos A B --out-csv outputs_eval_compare\full_eval_comparison.csv --out-json outputs_eval_compare\full_eval_comparison.json --out-md outputs_eval_compare\full_eval_comparison.md
if errorlevel 1 goto fail

echo.
echo Done. Check outputs_eval_compare\full_eval_comparison.md
goto end

:fail
echo.
echo Failed. Check the error above.

:end
pause
