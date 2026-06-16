# 换电脑运行指南

本项目的 GitHub 地址：https://github.com/roc-climate/weather-prob-model

---

## 方式一：git clone（推荐）

```bash
git clone git@github.com:roc-climate/weather-prob-model.git
cd weather-prob-model
```

> 如果 SSH 不通，用 HTTPS：`git clone https://github.com/roc-climate/weather-prob-model.git`

## 方式二：拷贝文件夹

将整个 `weather_prob_model` 文件夹拷贝到新电脑（U 盘 / 网盘 / 局域网）。

---

## 三步跑通

### 第 1 步：装 Python 环境

**Windows（conda）：**
```bash
conda create -n weather-prob python=3.12 -y
conda activate weather-prob
```

**Mac / Linux（venv）：**
```bash
python3 -m venv venv
source venv/bin/activate
```

**装 PyTorch + 依赖：**
```bash
# GPU 版（NVIDIA 显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU 版（无显卡 / Mac）
pip install torch torchvision torchaudio

# 项目依赖
pip install -r requirements.txt
```

**验证：**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from models.weather_model import WeatherProbModel; print('OK')"
```

### 第 2 步：准备数据

**方案 A — Mock 数据（最快，无需网络，几分钟生成）**
```bash
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014
```

**方案 B — 真实 ERA5 数据（需要 CDS 账号）**
```bash
# 1. 在 https://cds.climate.copernicus.eu/ 注册，获取 API key
# 2. 创建 ~/.cdsapirc 凭据文件
# 3. 下载（~500 MB，视网络可能需要较长时间）
python -m data.download_era5_monthly --years 1995 2019 --output ./data/raw/era5_monthly
python -m data.download_climate_indices --output ./data/raw/climate_indices
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014
```

### 第 3 步：训练

```bash
# GPU
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --output_dir ./checkpoints \
    --epochs 100 --device cuda

# CPU（慢，仅验证 pipeline）
python -m training.train --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly --output_dir ./checkpoints \
    --epochs 20 --device cpu
```

---

## 训练完成后

```bash
# 数值评估
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# Lead time 衰减评估
python -m evaluation.run_leadtime --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 生成图表
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly --years 2015 2016 \
    --output ./results --device cuda
```

查看结果：
- `checkpoints/history.json` — 训练曲线
- `results/` — 5 张评估图（空间 skill、RMSE、散点图、rank histogram、降水散点）

---

## 硬件参考

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU 显存 | 3 GB | 8 GB+ |
| 磁盘 | 2 GB | 10 GB |
| RAM | 8 GB | 16 GB+ |
| 系统 | Win/Mac/Linux | Linux |

20 epochs 训练：
- RTX 3090：~2 分钟
- Quadro P2200 (5GB)：~3 分钟
- CPU：~30 分钟

---

## 常见问题

**Q: 显存不够（CUDA out of memory）**
```bash
python -m training.train ... --batch_size 8
```

**Q: netCDF 读取出错**
报 `Unknown file format` 通常是 Windows 上 netCDF4 C 库问题。
用 mock 数据可避免（scipy 引擎）。或者删除旧 `.nc` 重新生成。

**Q: 训练 loss 是 NaN**
数据里有 NaN（SST 在陆地、土壤湿度在海洋）。
检查 `data/dataset.py` 是否有 `torch.nan_to_num(input_data, nan=0.0)`。

**Q: CDS 下载太慢**
CDS 在欧洲，国内直连 ~17 kB/s。建议先用 mock 数据验证 pipeline，
等网络好再下载真实数据。

**Q: Mac 上用不了 CUDA**
正常，Mac 没有 NVIDIA GPU。用 `--device cpu`（慢）或 MPS：`--device mps`。

**Q: 推送到 GitHub 失败**
- HTTPS 被 reset：换 SSH（`git remote set-url origin git@github.com:...`）
- SSH 也没通：换 Gitee 镜像

---

## 关于 .gitignore

以下内容**不会被提交到 GitHub**（太大或可重新生成）：
- `data/raw/` `data/processed/` — 数据文件
- `checkpoints/` — 训练好的模型
- `results/` — 评估图表

别人 clone 后需要自己跑 `mock_era5.py` 生成数据或 `train.py` 训练模型。
