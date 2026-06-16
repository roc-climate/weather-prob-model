"""Quick diagnostic: check data values driving spatial plots."""
import sys; sys.path.insert(0, '.')
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import WeatherDataset
from data.normalization import load_statistics
from models.weather_model import WeatherProbModel

variable_list = ['t2m','tp','sst','swvl1','sd','siconc','msl','u10','v10','tisr','ssr','str']
norm_stats = load_statistics('data/processed/norm_stats.json')
t2m_mean = norm_stats['mean'][0].item()
t2m_std = norm_stats['std'][0].item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
model = WeatherProbModel()
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()

ds = WeatherDataset(data_dir='data/raw/era5_monthly', variable_list=variable_list,
                     lead_time=1, years=(2015,2016), norm_stats=norm_stats)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

all_mu, all_sigma, all_y = [], [], []
with torch.no_grad():
    for batch in loader:
        pred = model(batch['x_atmos'].to(device), batch['x_slow'].to(device),
                     torch.zeros(1, 6, device=device))
        all_mu.append(pred['t2m']['mu'][0,0].cpu().numpy())
        all_sigma.append(pred['t2m']['sigma'][0,0].cpu().numpy())
        all_y.append(batch['y_t2m'][0,0].cpu().numpy())

mu = np.stack(all_mu) * t2m_std + t2m_mean      # (N, H, W)
sigma = np.stack(all_sigma) * t2m_std             # (N, H, W)
y = np.stack(all_y) * t2m_std + t2m_mean          # (N, H, W)

# Climatology
tds = WeatherDataset(data_dir='data/raw/era5_monthly', variable_list=variable_list,
                      lead_time=1, years=(1995,2014), norm_stats=norm_stats)
t_clim = tds.data[:, 0].numpy().mean(axis=0)  # already physical K  # (H,W)

# Per-grid RMSE
rmse_m = np.sqrt(np.mean((mu - y)**2, axis=0))
rmse_c = np.sqrt(np.mean((t_clim - y)**2, axis=0))

print(f'Model RMSE:  mean={rmse_m.mean():.2f}K  min={rmse_m.min():.2f}K  max={rmse_m.max():.2f}K')
print(f'Climo RMSE:  mean={rmse_c.mean():.2f}K  min={rmse_c.min():.2f}K  max={rmse_c.max():.2f}K')
print(f'Skill map:   mean={(rmse_m-rmse_c).mean():.2f}K  min={(rmse_m-rmse_c).min():.2f}K  max={(rmse_m-rmse_c).max():.2f}K')
print(f'mu range:    [{mu.min():.1f}, {mu.max():.1f}]')
print(f'y range:     [{y.min():.1f}, {y.max():.1f}]')
print(f'clim range:  [{t_clim.min():.1f}, {t_clim.max():.1f}]')
print(f'sigma range: [{sigma.min():.3f}, {sigma.max():.3f}] mean={sigma.mean():.3f}')

# RMSE by land vs ocean (from mock data)
mask = np.ones((121,240), dtype=bool)  # simple: all points
print(f'\nRMSE tropical (lat 20S-20N): model={rmse_m[40:80,:].mean():.2f}K  climo={rmse_c[40:80,:].mean():.2f}K')
print(f'RMSE midlat N (20-60N):       model={rmse_m[20:46,:].mean():.2f}K  climo={rmse_c[20:46,:].mean():.2f}K')
print(f'RMSE polar (>60):             model={rmse_m[:20,:].mean():.2f}K  climo={rmse_c[:20,:].mean():.2f}K')
