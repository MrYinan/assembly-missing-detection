#  Industrial Assembly Anomaly Detection System Based on Normal Feature Modeling

This project targets the quality inspection of assembly parts on industrial production lines. It designs and implements a continuous video inspection system based on visual anomaly detection to address issues such as missing end caps, structural anomalies, and local defects.

The system follows a **normal-only anomaly detection paradigm**: manual labels are not used during training. By learning the distribution of normal assembly parts in the deep feature space, it achieves the detection of unknown defects, avoiding the reliance of traditional supervised learning methods on large amounts of annotated anomaly data.

## System Overview
<img width="2816" height="1536" alt="Gemini_Generated_Image_scs6o9scs6o9scs6" src="https://github.com/user-attachments/assets/ecd42efe-4040-4801-b2ff-fc847293f087" />

The core methods include:

- Feature extraction using an ImageNet pre-trained ResNet18
- PatchCore-style local feature memory modeling
- Nearest neighbor distance anomaly scoring via Memory Bank
- Memory bank compression using Coreset Sampling
- Local neighborhood patch feature enhancement
- Pixel-level anomaly heatmap localization
- Video-level temporal consistency alarm strategy

## Installation
```bash
pip install -r requirements.txt
```
Note: If your CUDA environment is not configured, you can change the device parameter in the configuration file to cpu. The default setting is auto, which will automatically detect if CUDA is available.

## Run Video A：
```bat
scripts/run_videoA_deep_patchcore.bat
```

## Run Video B 
```bat
scripts\run_videoB_deep_patchcore.bat
```

## Generate Evaluation and Report Figures
```bat
scripts/analyze_metrics.bat
scripts/generate_report_figures.bat
```
