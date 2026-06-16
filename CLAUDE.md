# CLAUDE.md — Global 1.5° Probabilistic S2S Model

## 项目概述

全球 1.5° 概率性次季节天气预测深度学习模型。ConvNeXt + Cross-Attention 编码器，
Gaussian + 分位数概率头。聚焦 **3 周+ lead time** 的 t2m（温度）和 tp（降水）预测。

GitHub: https://github.com/roc-climate/weather-prob-model

## 架构速览

```
x_atmos (6ch) → ConvNeXtEncoder (3-stage) → z_atmos (1/8 分辨率)
x_slow  (4ch) → SlowVarEncoder  (2-stage) → z_slow  (1/4 分辨率)
x_index (6dim) → IndexEmbedding (MLP)      → z_idx   (64-dim vector)

z_slow 作为 query, z_atmos 作为 key/value → CrossAttentionFusion (4 layers)
z_fused + z_idx (spatially tiled) → GaussianHead → μ, σ (t2m)
                                  → QuantileHead → 5 quantiles (tp)
```

参数量: ~36M。20 epochs GPU 训练 < 5 分钟。

## 关键文件

| 文件 | 作用 | 特别说明 |
|------|------|---------|
| `configs/config.yaml` | 主配置 | `type` 字段已删除（会传入不认识的 kwargs）|
| `configs/variables.yaml` | 12 变量定义 | 6 组: targets, slow_forcing, atmos_state, external_forcing, energy_budget |
| `models/weather_model.py` | 顶层模型 | 组装所有子模块，默认 n_atmos_vars=6 |
| `models/encoder.py` | ConvNeXt 编码器 | 3 stage, 4x stem + 2x+2x+2x 下采样 |
| `models/slow_encoder.py` | 慢变量编码器 | 2 stage，输出更大空间尺寸 |
| `models/cross_attention.py` | 交叉注意力 | slow→atmos attention + 正弦位置编码 |
| `models/prob_head.py` | 概率头 | GaussianHead (μ,logvar) + QuantileHead (5 quantiles) |
| `data/dataset.py` | PyTorch Dataset | 加载 .nc，归一化 target，NaN→0 |
| `data/mock_era5.py` | ★ 模拟数据生成 | 无需网络，包含季节循环+ENSO+SST遥相关 |
| `data/normalization.py` | Z-score 统计 | 计算/保存/加载 norm_stats.json |
| `losses/crps_loss.py` | Gaussian CRPS | 解析形式，cos(lat) 加权 |
| `losses/quantile_loss.py` | Pinball loss | 多分位数 |
| `training/train.py` | ★ 训练主脚本 | warmup+cosine decay+AMP+checkpoint |
| `evaluation/run_eval.py` | ★ 评估入口 | CRPS/CRPSS/RMSE vs 气候态 baseline |
| `evaluation/run_leadtime.py` | ★ 多 lead time 评估 | 测试 skill 随 lead time 衰减 |
| `evaluation/run_plots.py` | ★ 可视化入口 | 空间图+散点图+rank histogram |
| `evaluation/metrics.py` | 指标函数 | 含 lat_weight bug 修复记录 |
| `evaluation/diagnose.py` | 数据诊断 | 检查空间图数值是否合理 |

## 工作流

```bash
# 1. 数据
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly

# 2. 归一化
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014

# 3. 训练
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --device cuda

# 4. 评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 5. Lead time 衰减
python -m evaluation.run_leadtime --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 6. 可视化
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

## 数据格式

- **文件**: 每年 `era5_monthly_surface_YYYY.nc`，包含 `valid_time × latitude × longitude`
- **网格**: 121×240 (1.5°)，纬度 -90→90，经度 0→358.5
- **变量顺序**: `[t2m, tp, sst, swvl1, sd, siconc, msl, u10, v10, tisr, ssr, str]`
- **targets**: t2m (idx=0) 和 tp (idx=1)
- **归一化**: `__getitem__` 中应用 z-score，mean/std 存于 `data/processed/norm_stats.json`
- **target 也归一化**: 训练时 target 被 `(y - mean)/std`，评估时反归一化

## 关键坑点（修复记录）

1. **`self.data` 存的是原始物理值**，不是归一化值。归一化仅在 `__getitem__` 中应用。
   评估脚本中读 `train_ds.data` 做气候态时，不要再次 `*std + mean`。

2. **SST 在陆地是 NaN，土壤湿度在海洋是 NaN** → `torch.nan_to_num(x, nan=0.0)` 在归一化后填充。
   归一化后均值=0，所以填 0 是合理的中性值。

3. **配置里的 `type` 字段必须删除** — config dict 直接作为 kwargs 传给 encoder/prob_head 的 `__init__`，
   不认识 `type` 会 TypeError。

4. **lat_weights 不要 squeeze** — `cos(lat)` 权重形状 (121,1)，squeeze 成 (121,) 后与 (121,240) 广播失败。

5. **读 .nc 不要指定 `engine="netcdf4"`** — Windows 上 C 库可能有问题。让 xarray 自动选后端。

6. **位置编码** — 原始实现有广播 bug（(1,W)×(dim//4,) 当 W≠dim//4 时失败）。
   重写为 meshgrid + unsqueeze 模式。

7. **dataset `_load_data`** — 原始实现逐变量逐文件循环导致维度错乱。
   改为每年 `(n_vars,T,H,W)` 再沿时间轴 concatenate。

8. **Slow encoder 和 Atmos encoder 输出不同空间尺寸** (1/4 vs 1/8) —
   交叉注意力做 sequence-to-sequence attention 可以处理不同的 L_q 和 L_kv，这是设计的意图。

## 评估结果（Mock 数据，20 epochs）

| Lead Time | CRPSS | 解读 |
|-----------|-------|------|
| 1 month (~4w) | +0.78 | 强技巧 |
| 2 months (~8w) | +0.34 | 部分技巧 |
| 3 months (~12w) | -0.34 | 无技巧 |

## 环境

- Python 3.12, PyTorch 2.x
- 依赖: `requirements.txt`（numpy, xarray, netCDF4, scipy, matplotlib, cartopy, pyyaml, cdsapi, pandas）
- GPU: CUDA 12.4 测试通过（Quadro P2200 5GB）
- 换电脑: `conda create -n weather-prob python=3.12 -y && conda activate weather-prob && pip install -r requirements.txt`
