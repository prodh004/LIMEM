[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prodh004/LIMEM/blob/main/LIMEM.ipynb)

# Learning Lake Ice Phenology and Freeze-Thaw Dynamics using L-band Radiometry: Insights from SMAP Satellite Observations

This repository presents a demonstration of lake ice phenology detection and a sequential detection–estimation framework for lake ice phenology characterization and brightness temperature (TB) simulation using SMAP L-band passive microwave observations over Great Bear Lake, Canada.

The notebook is organized into two main components. First, a non-parametric binary segmentation algorithm with a radial basis function (RBF) kernel detects four seasonal phenological breakpoints -- melt start, ice-off, freeze start, and ice-on -- from dual-polarized TB time series, further refined into three melt sub-stages (M1, M2, M3). Second, a sequential detection–estimation framework simulates annual dual-polarized TB time series by combining a supervised XGBoost classifier with state-specific LIMEM inversions.

---

## Methodology

### 1. Lake Ice Phenology Detection

The phenology detection algorithm identifies four seasonal breakpoints through a three-stage refinement pipeline. Binary segmentation (`ruptures.Binseg`) with an RBF kernel is applied to smoothed dual-polarized TB time series to locate approximate change points, which are anchored to local extrema and refined using the Mann-Whitney U test.

<p align="center">
  <img src="https://raw.githubusercontent.com/prodh004/LIMEM/main/Figure/Fig_01.jpeg" width="600"/>
</p>
<p align="center">
  <em>Figure 1. Schematics of the water–ice–snow continuum representing upward and downward fluxes of microwave radiation across the snow-air (sa), ice-snow (is), and water-ice (wi) interfaces, where T_s, T_i, and T_w are the respective physical temperatures of each layer.</em>
</p>

### 2. Sequential Detection–Estimation Framework

Annual TB time series are simulated using a sequential detection–estimation framework. A supervised XGBoost classifier trained on 2017–2024 SMAP observations identifies six phenological states (F1, M1, M2, M3, W, F0) using a 5-day rolling window of dual-polarized TBs. LIMEM is then inverted based on the detected state: roughness parameters are calibrated during F1; snow and ice wetness are jointly retrieved during M1 via Tikhonov-regularized inversion subject to w_s ≥ w_i; only ice wetness is retrieved during M2; fractional ice cover is retrieved via linear mixing during M3 and F0; and open-water TBs are computed analytically with daily roughness inversion.

---

## Data

| Variable | Description | Source |
|---|---|---|
| `TBh`, `TBv` | L-band brightness temperatures at H and V polarizations (1.4 GHz) | [SMAP L3 Enhanced](https://smap.jpl.nasa.gov/) |
| `lmlt` | Lake mixed-layer temperature (K) | [ERA5-Land](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) |
| `lict` | Lake ice surface temperature (K) | ERA5-Land |
| `licd` | Lake ice depth (m) | ERA5-Land |
| `t2m` | 2m air temperature (K) | ERA5-Land |
| `sd` | Snow depth over lake ice (m) | [Li et al. (2022)](https://doi.org/10.5194/tc-2022-42) |

---

## Code

### Setup

To run this notebook on Google Colab, clone this repository:

```python
!git clone https://github.com/prodh004/LIMEM.git
os.chdir("LIMEM")
```

### Key dependencies

```python
import numpy as np
import ruptures as rpt
from scipy.optimize import least_squares, minimize
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from xgboost import XGBClassifier
from LakeIceEmit import LakeIceEmit
```

### Phenology Detection

```python
# Detect four breakpoints per pixel using RBF kernel binary segmentation
bp = detect_change_points(tv_smooth, th_smooth, th_annual, tv_annual)
# Returns [BP0, BP1, BP2, BP3]: melt start, ice-off, freeze start, ice-on
```

### Detection–Estimation

```python
# Simulate annual TB time series
results = predict_annual_tb(tbh, tbv, lmlt, lict, licd, t2m, snow_depth,
                             final_model, LakeIceEmit, window=5)
# Returns tbh_pred, tbv_pred, classes, h_p, Q, m2_ws
```

---

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/prodh004/LIMEM/main/Figure/Results_01.jpg" width="700"/>
</p>
<p align="center">
  <em>L-band TB time series at H and V polarizations alongside Sentinel-2 false-color imagery capturing the freeze–thaw cycle of Great Bear Lake (66.42°N, 121.23°W).</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/prodh004/LIMEM/main/Figure/Results_02.jpg" width="700"/>
</p>
<p align="center">
  <em>Lake ice phenology retrievals for Great Bear Lake in 2018 comparing SMAP (i), IMS (ii), Sentinel-2 (iii), and MODIS (iv) — frozen (white), melting (blue), open-water (black), freezing (gray).</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/prodh004/LIMEM/main/Figure/Results_03.jpg" width="700"/>
</p>
<p align="center">
  <em>Observed and simulated polarized TB time series for five freshwater lakes during 2016, with shaded regions indicating detected phenological states (F1, M1, M2, M3, W, F0).</em>
</p>
