# CLAUDE.md

## 项目概述

全球 1.5° 概率性 S2S（次季节到季节）天气预报深度学习模型。
核心聚焦 **3 周+ lead time** 的 t2m（温度）和 tp（降水）预测。

## 架构速览

```
输入 x_atmos (6ch) → ConvNeXtEncoder → z_atmos
输入 x_slow  (4ch) → SlowVarEncoder  → z_slow
输入 x_index (6维) → IndexEmbedding  → z_idx

z_slow (query) + z_atmos (key/value) → CrossAttentionFusion → z_fused
z_fused + z_idx → GaussianHead  → μ, σ (t2m)
                → QuantileHead  → 5个分位数 (tp)
```

- **AtmosEncoder**: ConvNeXt，3 stage，每个 stage 2x 下采样 → 输出 1/8 分辨率
- **SlowEncoder**: 更轻的 ConvNeXt，2 stage → 1/4 分辨率
- **CrossAttention**: slow 特征作为 query 关注 atmos 特征，建模遥相关
- **GaussianHead**: 输出 μ 和 log σ，CRPS 解析损失
- **QuantileHead**: 输出 [0.1, 0.25, 0.5, 0.75, 0.9] 分位数，pinball loss

## 关键文件

| 文件 | 作用 |
|------|------|
| `configs/config.yaml` | 主配置：模型结构、训练参数、数据路径 |
| `configs/variables.yaml` | 12 个变量的定义、分组、归一化参数 |
| `models/weather_model.py` | 顶层模型，组装所有子模块 |
| `models/encoder.py` | ConvNeXt 大气编码器 |
| `models/cross_attention.py` | 交叉注意力融合 + 位置编码 |
| `models/prob_head.py` | Gaussian 头和分位数头 |
| `data/dataset.py` | PyTorch Dataset，load .nc 文件，归一化 |
| `data/mock_era5.py` | **生成本地模拟数据**（无网测试用）|
| `losses/crps_loss.py` | Gaussian CRPS 解析形式 + cos(lat) 加权 |
| `losses/quantile_loss.py` | 分位数 pinball loss |
| `training/train.py` | 训练主脚本（含 warmup + cosine decay）|
| `evaluation/run_eval.py` | 评估入口：CRPS/CRPSS/RMSE vs 气候态 baseline |
| `evaluation/run_plots.py` | 可视化入口：空间图、散点图、rank histogram |
| `evaluation/metrics.py` | 指标函数库 |
| `evaluation/baselines.py` | 气候态/持续性 baseline |

## 工作流

```bash
# 1. 数据（二选一）
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly  # 模拟
# 或
python -m data.download_era5_monthly --years 1995 2019 ...                     # 真实 CDS

# 2. 归一化
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014

# 3. 训练
python -m training.train --config configs/config.yaml --data_dir ./data/raw/era5_monthly --device cuda

# 4. 评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 5. 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

## 数据格式

- **输入**: 每年一个 `era5_monthly_surface_YYYY.nc`，包含 `valid_time × latitude × longitude` 的 12 个变量
- **网格**: 121×240（1.5°），纬度 -90 到 90，经度 0 到 358.5
- **变量顺序**: `[t2m, tp, sst, swvl1, sd, siconc, msl, u10, v10, tisr, ssr, str]`
- **target**: t2m 和 tp（前两个变量）
- **归一化**: 在 `dataset.py.__getitem__` 中做 z-score，mean/std 存在 `data/processed/norm_stats.json`

## 注意

- `self.data` 存的是**原始物理值**（未归一化），归一化仅在 `__getitem__` 返回时应用
- SST 在陆地是 NaN，土壤湿度在海洋是 NaN → `torch.nan_to_num(x, nan=0.0)` 填充
- 配置里的 `type` 字段已删除（会导致 `__init__` 收到不认识的 kwargs）
- 读取 .nc 文件不要指定 `engine="netcdf4"`，让 xarray 自动选后端
- 训练时 target 也被归一化了（模型输出归一化空间的值），评估时需反归一化
- loss 的纬度加权：`cos(lat)` 权重，形状 (121, 1)，不要 squeeze 最后一维
