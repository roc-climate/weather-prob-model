# Global 1.5° Probabilistic S2S Weather Prediction Model

A deep learning model for probabilistic subseasonal-to-seasonal (S2S) weather prediction
at 1.5° resolution with weekly time steps, **focused on week-3+ lead times**.

## 科学动机

3 周以上天气预报的可预测性来源从大气初始条件迁移到**慢变边界强迫**：
海表温度（SST）、土壤湿度、积雪/海冰、季节辐射循环。MJO 和 ENSO 等气候模态
提供了额外的 3-4 周预测信号。

常规 NWP 模型在这个时间尺度上的技巧迅速衰减。本项目的目标是利用深度学习
直接从慢变变量中学习遥相关和边界强迫驱动的预测信号。

## 预测目标和变量

### 预测目标
| 变量 | 说明 | 概率头 |
|------|------|--------|
| **t2m** | 2 米近地面温度 | Gaussian（CRPS 损失） |
| **tp** | 总降水量 | 分位数回归（Pinball 损失） |

### 输入变量（12 个 surface-only）
| 分组 | 变量 | 说明 |
|------|------|------|
| **慢变强迫** | sst, swvl1, sd, siconc | SST、土壤湿度、雪深、海冰覆盖 |
| **大气状态** | msl, u10, v10 | 海平面气压、10 米风 |
| **辐射/能量** | tisr, ssr, str | 大气顶/地表短波、地表长波辐射 |

> 刻意不包含 pressure-level 变量：对流层中上层的初始条件信息在 3 周后几乎完全丢失。

## 模型架构

```
输入:
  x_atmos (6 通道)  ──→  AtmosEncoder (ConvNeXt)  ──→  z_atmos
  x_slow  (4 通道)  ──→  SlowEncoder  (ConvNeXt)  ──→  z_slow
  x_index (6 标量)   ──→  IndexEmbedding (MLP)    ──→  z_idx

  z_slow + z_atmos  ──→  CrossAttentionFusion  ──→  z_fused
  z_fused + z_idx   ──→  GaussianHead  ──→  μ, σ (t2m)
                      ──→  QuantileHead  ──→  [q₀.₁, q₀.₂₅, q₀.₅, q₀.₇₅, q₀.₉] (tp)
```

### 关键设计选择

- **ConvNeXt 编码器**：轻量卷积，提取大气和慢变量的空间特征
- **交叉注意力融合**：显式建模遥相关（热带 SST → 中纬度温度），CNN 感受野不够
- **气候指数嵌入**：MJO/ENSO/NAO/AO 作为标量条件信号直接注入
- **独立概率头**：t2m 用 Gaussian CRPS，降水用分位数回归（偏态分布）
- **参数量**：~36M，单 GPU 轻松训练

## 数据

### 数据源
| 数据 | 来源 | 说明 |
|------|------|------|
| ERA5 月均 surface | CDS API | 12 变量，0.25°→1.5°，~500MB/25年 |
| MJO RMM 指数 | BOM Australia | 日尺度，30-90 天周期 |
| Niño 3.4 指数 | NOAA CPC | ENSO 监测 |
| NAO/AO 指数 | NOAA CPC | 中纬度环流模态 |

### 数据集划分
| 时期 | 年份 | 月数 | 用途 |
|------|------|------|------|
| 训练 | 1995–2014 | 240 | 模型训练 |
| 验证 | 2015–2016 | 24 | 早停和超参数选择 |
| 测试 | 2017–2019 | 36 | 最终评估 |

### Mock 数据（验证用）
`data/mock_era5.py` 可生成本地模拟数据进行 pipeline 测试，
无需网络下载。模拟数据包含：
- 季节性温度循环（南北半球反相）
- Niño 3.4 型热带 SST 异常
- SST → t2m 遥相关信号（1 个月滞后）
- 偏态降水分布、极地雪/冰变量

## 训练

```bash
# 完整训练（GPU）
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly \
    --output_dir ./checkpoints \
    --epochs 100 --device cuda
```

### 训练配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | lr=1e-3, weight_decay=1e-5 |
| 学习率策略 | Cosine annealing | 5 epoch 线性 warmup |
| 批量大小 | 32 | |
| Epochs | 100 | Phase 1: 单 lead time |
| 混合精度 | AMP (CUDA 可选) | |
| 纬度加权 | cos(lat) | 面积加权平均 |

### 损失函数
- **t2m**：Gaussian CRPS（解析形式，无需采样）
- **tp**：分位数 Pinball loss（τ = 0.1, 0.25, 0.5, 0.75, 0.9）

### Lead time 阶段
| 阶段 | Epochs | Lead Times | 说明 |
|------|--------|------------|------|
| Phase 1 | 1-50 | 1 个月 | 单 lead time 训练 |
| Phase 2 | 51-80 | 1,2,3 个月 | 多 lead time 联合 |
| Phase 3 | 81-100 | 1,2,3 个月 | 微调 + 早停 |

## 评估

```bash
# 数值评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

### 评估指标
| 指标 | 说明 | 目标 |
|------|------|------|
| **CRPS** | 连续分级概率评分（主排名指标） | 越小越好 |
| **CRPSS** | CRPS 技巧评分（vs 气候态） | >0 表示优于气候态 |
| **RMSE** | 均方根误差（确定性） | 辅助参考 |
| **ACC** | 异常相关系数 | >0.6 为有用 |
| **Rank Histogram** | 校准诊断 | 平坦 = 良好校准 |
| **Spread-Skill Ratio** | 集合散度 vs 误差 | ≈1.0 理想 |

### Baseline 对比
1. **气候态**：用训练期月平均气候态作为预测
2. **持续性**：假设异常持续
3. **阻尼持续性**：异常指数衰减

### Mock 数据验证结果（20 epochs）
| 指标 | 模型 | 气候态 Baseline |
|------|------|----------------|
| t2m CRPS | **0.61 K** | 2.79 K |
| t2m RMSE | **1.10 K** | 5.00 K |
| t2m CRPSS | **+0.78** | — |
| t2m R² | **0.90** | — |

## 项目结构

```
weather_prob_model/
├── configs/
│   ├── config.yaml              # 主配置（模型、训练、评估）
│   └── variables.yaml           # 变量定义和分组
├── data/
│   ├── download_era5_monthly.py # CDS API 下载（月均数据）
│   ├── download_climate_indices.py # MJO/ENSO/NAO/AO 下载
│   ├── mock_era5.py             # 本地模拟数据生成（测试用）
│   ├── dataset.py               # PyTorch Dataset
│   ├── normalization.py         # Z-score 归一化
│   └── weekly_aggregate.py      # 日→周聚合（Phase 2）
├── models/
│   ├── encoder.py               # ConvNeXt 大气编码器
│   ├── slow_encoder.py          # 慢变量编码器
│   ├── cross_attention.py       # 交叉注意力遥相关融合
│   ├── index_embedding.py       # 气候指数嵌入
│   ├── prob_head.py             # Gaussian + 分位数头
│   └── weather_model.py         # 完整模型
├── losses/
│   ├── crps_loss.py             # 解析 Gaussian CRPS
│   └── quantile_loss.py         # 分位数 Pinball loss
├── training/
│   ├── train.py                 # 训练主脚本
│   └── scheduler.py             # LR 调度 + 课程学习
├── evaluation/
│   ├── metrics.py               # CRPS/CRPSS/RMSE/ACC/校准
│   ├── baselines.py             # 气候态/持续性 baseline
│   ├── calibration.py           # Rank Histogram/PIT/校准
│   ├── plot_results.py          # 空间图/散点图/直方图
│   ├── run_eval.py              # 评估入口
│   ├── run_plots.py             # 可视化入口
│   └── diagnose.py              # 数据诊断
├── scripts/
│   ├── run_download.sh          # 一键下载
│   ├── run_train.sh             # 训练启动
│   └── run_eval.sh              # 评估启动
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成 mock 数据（跳过网络下载，验证 pipeline）
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly

# 3. 计算归一化统计量
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014

# 4. 训练（GPU 推荐，CPU 也可）
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --epochs 20 --device cuda

# 5. 评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 6. 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

> 用真实 ERA5 数据时，将步骤 2 替换为 `bash scripts/run_download.sh`（需要 CDS 账号）。

## 后续升级路线

- **Phase 2**：日→周聚合 + 真实气候指数 + 多 lead time 联合训练
- **Phase 3**：Gaussian → Conditional Flow Matching（CFM）概率升级
- **Phase 4**：后处理校准 + 极端事件评估 + 可解释性分析

## 参考文献

- Vitart et al. (2017) "The Subseasonal to Seasonal (S2S) Prediction Project"
- Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
- Liu et al. (2022) "ConvNeXt" — encoder backbone
- Lipman et al. (ICML 2023) "Conditional Flow Matching"
- GenCast (Price et al., 2024) — probabilistic weather prediction
