@echo off
cd /d %~dp0\..
python -m src.debug_rois --video-id B --video data\videoB_new.mp4 --config configs\videoB_deep_patchcore.json --times 651 800 1000 1048 1053 1073 1079 1086 --out-dir figures\debug_rois
pause
