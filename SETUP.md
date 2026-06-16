# 换电脑运行指南

## 你需要拷贝的东西

**只需要拷贝 `weather_prob_model` 这一个文件夹。** 里面包含了所有代码、配置、数据（mock 数据 ~660MB）和训练好的模型。

拷贝方式任选其一：
- U 盘 / 移动硬盘直接拷文件夹
- 上传到 GitHub / GitLab，新电脑 `git clone` 下来
- 网络共享 / 云盘传压缩包

---

## 新电脑上只需 3 步

### 第 1 步：装 Python 环境

打开终端（Windows 用 PowerShell，Mac/Linux 用 Terminal），进入项目目录：

```bash
cd weather_prob_model
```

**Windows（推荐 conda）：**
```bash
conda create -n weather-prob python=3.12 -y
conda activate weather-prob
```

**Mac / Linux：**
```bash
python3 -m venv venv
source venv/bin/activate
```

然后装 PyTorch 和依赖：

```bash
# GPU 版（有 NVIDIA 显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU 版（无显卡 / Mac）
pip install torch torchvision torchaudio

# 项目依赖（不管 GPU 还是 CPU 都要装）
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from models.weather_model import WeatherProbModel; print('OK')"
```

看到 `OK` 就说明环境好了。如果最后一步报错，通常是少装了依赖，`pip install -r requirements.txt` 重新跑一遍。

### 第 2 步：准备数据

**情况 A — 你拷贝的文件夹里已经有数据**

`data/raw/era5_monthly/` 里有 `.nc` 文件 + `data/processed/` 里有 `norm_stats.json`，直接跳到第 3 步训练，什么都不用做。

**情况 B — 文件夹里没有数据（从 git clone 的，或者只拷了代码）**

几分钟本地生成 mock 数据：

```bash
python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014
```

如果需要真实 ERA5 数据（需要 CDS 账号）：

```bash
# 1. 在 https://cds.climate.copernicus.eu/ 注册
# 2. 创建凭据文件 ~/.cdsapirc
# 3. 下载
python -m data.download_era5_monthly --years 1995 2019 --output ./data/raw/era5_monthly
python -m data.download_climate_indices --output ./data/raw/climate_indices
python -m data.normalization --data_dir ./data/raw/era5_monthly --years 1995 2014
```

### 第 3 步：训练

```bash
# 有 GPU
python -m training.train --config configs/config.yaml --data_dir ./data/raw/era5_monthly --output_dir ./checkpoints --epochs 100 --device cuda

# 无 GPU
python -m training.train --config configs/config.yaml --data_dir ./data/raw/era5_monthly --output_dir ./checkpoints --epochs 20 --device cpu
```

---

## 训练完评估

```bash
# 数值指标
python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda

# 生成图表
python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results --device cuda
```

---

## 硬件参考

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU 显存 | 3 GB | 8 GB+ |
| 磁盘 | 2 GB | 10 GB |
| RAM | 8 GB | 16 GB+ |
| 系统 | Win/Mac/Linux | Linux |

20 epochs 训练时间参考：
- GPU（RTX 3090）：~5 分钟
- GPU（Quadro P2200）：~3 分钟
- CPU：~30 分钟

---

## 常见问题

**显存不够（CUDA out of memory）**
```bash
python -m training.train ... --batch_size 8
```

**netCDF 读取出错**
报 `Unknown file format` 通常是 Windows 上 netCDF4 C 库问题。用 mock 数据（scipy 写的）可避免。或者删除旧 `.nc` 文件重新生成 mock 数据。

**训练 loss 是 NaN**
数据里有 NaN（SST 在陆地、土壤湿度在海洋）。确认 `data/dataset.py` 里有 `torch.nan_to_num(input_data, nan=0.0)` 这一行。

**下载 CDS 数据慢**
CDS 在欧洲，国内直连只有十几 KB/s。先用 mock 数据验证，等网络好再下载真实数据。

**Mac 上用不了 CUDA**
正常，Mac 没有 NVIDIA GPU。用 `--device cpu` 训练，或者用 MPS（Apple Silicon）：`--device mps`。
