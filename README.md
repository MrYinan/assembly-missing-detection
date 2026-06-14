# 基于正常特征建模的工业装配件少样本异常检测系统

本项目面向工业流水线装配件质量检测场景，针对端盖缺失、结构异常、局部缺陷等问题，设计并实现了一套基于视觉异常检测的连续视频检测系统。

系统采用**仅依赖正常样本的异常检测范式**，通过学习正常装配件在深度特征空间中的分布，实现未知缺陷检测，避免传统监督学习方法对大量异常标注数据的依赖。

核心方法基于：

- ImageNet 预训练 ResNet18 特征提取
- PatchCore 风格局部特征记忆建模
- Memory Bank 最近邻距离异常评分
- Coreset sampling 特征库压缩
- 局部邻域 Patch 特征增强
- 像素级异常热力图定位
- 视频级时序一致性报警策略


## 安装依赖
```bash
pip install -r requirements.txt
```
如果显卡 CUDA 环境没配好，可以把配置文件里的 `device` 改成 `cpu`；默认 `auto` 会自动检测 CUDA 是否可用。

## 运行 Video A：
```bat
scripts/run_videoA_deep_patchcore.bat
```

## 运行 Video B 
```bat
scripts\run_videoB_deep_patchcore.bat
```

## 生成评估和报告图片：
```bat
scripts/analyze_metrics.bat
scripts/generate_report_figures.bat
```
