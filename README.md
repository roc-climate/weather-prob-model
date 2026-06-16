# Global 1.5° Probabilistic S2S Weather Prediction Model

深度学习全球概率性次季节到季节（S2S）天气预报模型。1.5° 分辨率，月步长（Phase 2 升级为周步长），**聚焦第 3 周及以上的 lead time**。

> GitHub: https://github.com/roc-climate/weather-prob-model

---

## 科学动机

3 周以上可预测性的来源从大气初始条件迁移到**慢变边界强迫**：
SST、土壤湿度、积雪/海冰、季节辐射。MJO 和 ENSO 提供了额外的 3-4 周预测信号。

- 3 周内：大气初始条件主导 → 传统 NWP 仍有技巧
- 3-8 周：慢变变量主导 → SST 记忆 + MJO/ENSO 遥相关
- 8 周+：可预测性极低，气候态是强 baseline

本项目的核心假设：直接让模型从慢变变量 + 气候指数中学习遥相关，
不再依赖中间大气状态（不包含任何 pressure-level 变量）。

---

## 预测目标

| 变量 | 单位 | 概率头 | 损失函数 |
|------|------|--------|---------|
| **t2m** 2 米温度 | K | Gaussian (μ, σ) | CRPS（解析梯度）|
| **tp** 总降水量 | m | 分位数 (τ=0.1,0.25,0.5,0.75,0.9) | Pinball loss |

## 输入变量（12 个，全部 surface）

| 分组 | 变量 | 理由 |
|------|------|------|
| **慢变强迫** (4) | sst, swvl1, sd, siconc | SST 是 3 周+ 可预测性第一来源；土壤湿度/雪/海冰有长记忆 |
| **大气状态** (3) | msl, u10, v10 | 大尺度环流表征 |
| **辐射/能量** (3) | tisr, ssr, str | 季节循环 + 能量收支 |
| **气候指数** (6) | RMM1, RMM2, Nino3.4, NAO, AO, doy | MJO/ENSO 是已知的 S2S 预测关键因子 |

> 刻意不包含 pressure-level 变量（850/500/200 hPa 的 z,t,q,u,v）：对流层中上层
> 的初始条件信息在 3 周后几乎全部丢失，省下的数据量用于更长的时间序列。

---

## 模型架构

```
x_atmos (6ch)  ──→  AtmosEncoder (ConvNeXt)   ──→  z_atmos
x_slow  (4ch)  ──→  SlowEncoder  (ConvNeXt)   ──→  z_slow
x_index (6dim) ──→  IndexEmbedding (MLP)      ──→  z_idx

z_slow (query) + z_atmos (key/value)  ──→  CrossAttentionFusion  ──→  z_fused
z_fused + z_idx  ──→  GaussianHead   ──→  μ, σ      (t2m)
                 ──→  QuantileHead   ──→  5 个分位数 (tp)
```

### 设计选择

| 选择 | 理由 |
|------|------|
| **SST 单独编码通道** | 3 周+ 可预测性的核心载体，应走专用 encoder |
| **Cross-Attention 而非纯 CNN** | 遥相关是非局部的（热带太平洋 SST → 北美温度），CNN 需要很深才能有足够感受野 |
| **气候指数直接注入** | MJO/ENSO 是已知的 3-4 周预测因子，作为条件信号比让模型从网格中学要高效得多 |
| **Gaussian CRPS（解析）** | 无需采样，有解析梯度，训练快速 |
| **分位数回归（降水）** | 降水分布偏态严重，无分布假设 |
| **无 pressure-level 变量** | 减少数据量 + 避免模型依赖已失效的初始条件 |

### 参数量

~36M，单 GPU 轻松训练。AtmosEncoder ~15M、SlowEncoder ~5M、CrossAttention ~3M、Decoder heads ~10M。

---

## 数据

### 数据源

| 数据 | 来源 | 大小 | 说明 |
|------|------|------|------|
| ERA5 月均 surface | [CDS API](https://cds.climate.copernicus.eu/) | ~500 MB/25年 | 12 变量，0.25°→1.5° regrid |
| MJO RMM 指数 | [BOM Australia](http://www.bom.gov.au/climate/mjo/) | KB 级 | 日尺度，Phase 2 使用 |
| Niño 3.4 指数 | [NOAA CPC](https://www.cpc.ncep.noaa.gov/) | KB 级 | ENSO 监测 |
| NAO/AO 指数 | [NOAA CPC](https://www.cpc.ncep.noaa.gov/) | KB 级 | 中纬度环流模态 |

### 数据集划分

| 时期 | 年份 | 样本数 | 用途 |
|------|------|--------|------|
| 训练 | 1995–2014 | 240 月 | 模型训练 + 归一化统计 |
| 验证 | 2015–2016 | 24 月 | 早停 |
| 测试 | 2017–2019 | 36 月 | 最终评估 |

### Mock 数据

`data/mock_era5.py` — 本地生成模拟数据，**无需网络下载**，几分钟跑通全流程。
模拟了：季节循环、Niño 3.4 异常、SST→t2m 遥相关（1 月滞后）、偏态降水、极地冰雪。
用于在下载真实数据前验证 pipeline。

---

## 训练

```bash
# 一键训练
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --output_dir ./checkpoints \
    --epochs 100 --device cuda
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW, lr=1e-3, weight_decay=1e-5 |
| LR 调度 | 5 epoch 线性 warmup → cosine decay 到 1e-5 |
| Batch size | 32 |
| Epochs | 100 |
| 混合精度 | AMP（CUDA 自动启用）|
| 纬度加权 | cos(lat) 面积加权 |
| 损失权重 | t2m=1.0, tp=0.5 |
| 归一化 | Z-score（训练集统计，target 同步归一化）|

### 课程学习

| 阶段 | Epochs | Lead Times | 说明 |
|------|--------|------------|------|
| Phase 1 | 1-50 | 1 个月 | 单 lead time，lr=1e-3 |
| Phase 2 | 51-80 | 1,2,3 个月 | 多 lead time 联合，lr=5e-4 |
| Phase 3 | 81-100 | 1,2,3 个月 | 微调 + 验证早停 |

---

## 评估

```bash
# 完整评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 多 lead time 衰减评估
python -m evaluation.run_leadtime --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda

# 数据诊断
python -m evaluation.diagnose
```

### 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **CRPS** | Continuous Ranked Probability Score | 主排名指标，越低越好 |
| **CRPSS** | CRPS Skill Score | vs 气候态 baseline，>0 = 有技巧 |
| **RMSE** | Root Mean Square Error | 确定性参考（ensemble mean）|
| **ACC** | Anomaly Correlation Coefficient | 异常场相关，>0.6 为"有用" |
| **Rank Histogram** | — | 校准诊断：平坦 = 完美校准 |
| **Spread-Skill Ratio** | — | 集合散度 vs 误差比，≈1 理想 |

### Baseline 对比

1. **气候态**：用训练期月平均气候态预测（"这个月长什么样就预测什么样"）⭐ 最强 baseline
2. **持续性**：假设异常持续（"这月偏暖 2°C → 下月也偏暖 2°C"）
3. **阻尼持续性**：异常以 e-folding 时间衰减

### 验证结果（Mock 数据，20 epochs）

| 指标 | 模型 | 气候态 Baseline |
|------|------|----------------|
| t2m CRPS | **0.61 K** | 2.79 K |
| t2m RMSE | **1.10 K** | 5.00 K |
| t2m CRPSS | **+0.78** | — |
| t2m R² | **0.90** | — |

### Lead-Time 衰减（Mock 数据，mono-lead 模型）

| Lead Time | 对应约 | Model CRPS | Clim CRPS | **CRPSS** |
|-----------|--------|-----------|-----------|-----------|
| 1 个月 | **~4 周** | 0.61 K | 2.78 K | **+0.78** |
| 2 个月 | **~8 周** | 0.93 K | 2.78 K | **+0.34** |
| 3 个月 | **~12 周** | 3.74 K | 2.78 K | **−0.34** |

> 趋势：第 4 周模型有强技巧（CRPSS +0.78），第 8 周技巧减半（+0.34），
> 第 12 周技巧消失（−0.34）。与 "SST 记忆效应 4-8 周" 的科学预期一致。
> 注意：当前为单 lead time 训练的模型做 off-target 评估；Phase 2 多 lead time
> 联合训练后各 lead time 的 CRPSS 会更高。

---

## 项目结构

```
weather_prob_model/
├── configs/
│   ├── config.yaml               # 主配置
│   └── variables.yaml            # 12 变量定义、分组、归一化参数
├── data/
│   ├── mock_era5.py              # ★ 本地模拟数据生成（无网测试用）
│   ├── download_era5_monthly.py  # CDS API 下载
│   ├── download_climate_indices.py
│   ├── dataset.py                # PyTorch Dataset
│   ├── normalization.py          # Z-score 归一化
│   └── weekly_aggregate.py       # 日→周聚合（Phase 2）
├── models/
│   ├── weather_model.py          # 顶层模型
│   ├── encoder.py                # ConvNeXt 大气编码器
│   ├── slow_encoder.py           # 慢变量编码器
│   ├── cross_attention.py        # Cross-Attention + 位置编码
│   ├── index_embedding.py        # 气候指数嵌入
│   └── prob_head.py              # Gaussian + 分位数概率头
├── losses/
│   ├── crps_loss.py              # 解析 Gaussian CRPS + cos(lat) 加权
│   └── quantile_loss.py          # Pinball loss
├── training/
│   ├── train.py                  # ★ 训练主脚本
│   └── scheduler.py              # LR 调度 + 课程学习
├── evaluation/
│   ├── run_eval.py               # ★ 评估入口（CRPS/CRPSS/RMSE vs baseline）
│   ├── run_leadtime.py           # ★ 多 lead time 衰减评估
│   ├── run_plots.py              # ★ 可视化入口（空间图/散点图/rank histogram）
│   ├── metrics.py                # 指标函数库
│   ├── baselines.py              # 气候态/持续性 baseline
│   ├── calibration.py            # 校准诊断
│   ├── plot_results.py           # 绘图函数
│   └── diagnose.py               # 数据诊断
├── scripts/
│   ├── run_download.sh
│   ├── run_train.sh
│   └── run_eval.sh
├── requirements.txt
├── .gitignore
├── README.md / README.html       # 本文档
├── SETUP.md / SETUP.html         # ★ 新电脑复现指南
└── CLAUDE.md / CLAUDE.html       # Claude Code 项目速览
```

---

## 快速开始

```bash
# 1. 环境
conda create -n weather-prob python=3.12 -y && conda activate weather-prob
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. 数据（二选一）
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly     # mock
# python -m data.download_era5_monthly --years 1995 2019 ...                     # 真实 CDS

# 3. 归一化
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014

# 4. 训练
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --device cuda

# 5. 评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 6. 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

---

## 换电脑运行

```bash
git clone git@github.com:roc-climate/weather-prob-model.git
cd weather-prob-model
# 然后按上方"快速开始"的步骤走
```

详见 [SETUP.html](SETUP.html)。

---

## 硬件

| 配置 | 已测试 | 推荐 |
|------|--------|------|
| GPU | Quadro P2200 (5 GB) | 8 GB+ |
| 磁盘 | 2 GB (mock) / 6 GB (真实) | 10 GB |
| RAM | 16 GB | 16 GB+ |
| 训练时间 (20 epochs) | GPU ~3 min / CPU ~30 min | — |

---

## 升级路线

- **Phase 2**：日→周聚合 + 真实气候指数 + 多 lead time 联合训练 → 精确到周尺度的 CRPSS 衰减
- **Phase 3**：Gaussian → Conditional Flow Matching（CFM）→ 可以采样任意形状的概率分布
- **Phase 4**：后处理校准 + 极端事件评估 + SST 遥相关模式的可解释性分析

## 关键参考

- Vitart et al. (2017) — S2S Prediction Project
- Gneiting & Raftery (2007) — Strictly Proper Scoring Rules
- Liu et al. (2022) — ConvNeXt
- Lipman et al. (2023) — Conditional Flow Matching
- Price et al. (2024) — GenCast, probabilistic weather prediction
