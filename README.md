# 基于 ResNet-PatchCore 的工业装配件缺失检测 v16

本项目是大作业 03 的深度异常检测版本。核心检测方法采用 **ImageNet 预训练 ResNet + PatchCore Memory Bank + 最近邻异常分数**。OpenCV 只用于视频读取、ROI 定位和结果可视化，不作为主检测模型。

## 方法特点

- 训练阶段只读取正常装配视频段；
- ROI 裁剪后送入 ResNet18 提取 patch-level 深度特征；
- 建立正常 Memory Bank；
- 测试阶段计算当前 ROI patch 到正常库的最近邻距离；
- 人工异常时间点只用于最终复核，不参与训练、阈值设定和推理；
- 新增非作弊式时序后处理：先按 Deep PatchCore 分数生成候选报警，再按连续性与强异常分数过滤短时反光/过渡误报。

## 文件结构

```text
configs/                    # Video A / Video B 配置
src/deep_patchcore.py        # ResNet 特征提取 + Memory Bank
src/detectors.py             # Video A / Video B 检测器
src/pipeline.py              # 训练、完整测试段推理、视频渲染
src/evaluate_keyframes.py    # 人工复核点评价，只在推理后使用
src/generate_figures.py      # 完整测试段曲线和报警时间轴
labels/                      # 人工复核标签
scripts/                     # 一键运行脚本
```

## 放置视频
把完整 18 分 11 秒 Video B 放到：
```text
data/videoB_new.mp4
```

把 Video A 放到：
```text
data/videoA.mp4
```

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