# Analysis Functions for Thermal Memory Research

This directory contains the Python functions and libraries used for the analysis presented in the Nature Climate Change paper:

**"Observed multi-decadal increase in the surface ocean’s thermal inertia"**  
[https://www.nature.com/articles/s41558-025-02245-w](https://doi.org/10.1038/s41558-025-02245-w)

All figures and analysis results in the study, except for Mixed Layer Depth data, were created using the functions in this directory. The mixed layer depth data is from [Sallée et al. (2021)](https://doi.org/10.1038/s41586-021-03303-x).

## Overview

The analysis functions implement various methods to calculate ocean thermal memory, heat flux feedback rates, marine heatwave characteristics, and their long-term trends. These analyses help understand how the upper ocean's thermal memory is changing in response to climate change.

## Files and Functions

### 1. `thermal_memory_ar1.py`
**Purpose**: Calculate surface thermal memory using Autoregressive-1 (AR1) like model (it selects lag-10 autocorrelation coefficient to fit to exponential function)

**Key Functions**:
- `memory_timescale(da, vars, option='annual')`: Calculates thermal memory timescale from SST anomalies using autocorrelation
- `trend_test(x)`: Performs Mann-Kendall trend test for statistical significance

**Scientific Context**: The AR1 model assumes that the ocean's thermal memory follows an exponential decay process, allowing estimation of characteristic timescales.

### 2. `heatflux_feedback.py`
**Purpose**: Calculate negative heat flux feedback rates (λ_a and λ_o) using bulk formulas

**Key Functions**:
- `stability(t2m, sst)`: Calculates atmospheric stability parameter
- `vfunc_Ch(C_d, zeta)`: Computes heat transfer coefficient based on stability
- `calc_sh(C_h, wind10)`: Calculates sensible heat flux feedback
- `calc_lh(C_e, wind10, t2m)`: Calculates latent heat flux feedback
- `trend_test(x)`: Mann-Kendall trend test
- `Theilsen_test(y)`: Theil-Sen slope estimation

**Scientific Context**: Heat flux feedback represents the ocean's ability to dampen temperature anomalies through atmospheric heat exchange (λ_a) and oceanic processes (λ_o).

### 3. `annual_MHW.py`
**Purpose**: Analyze marine heatwave (MHW) duration trends from daily SST data

**Key Functions**:
- `TS_for_MHWduration(sst)`: Calculates MHW duration trends using Theil-Sen method
- `trend_test(x)`: Mann-Kendall trend test for MHW duration changes
- `main(lat_idx)`: Parallel processing function for global MHW analysis

**Scientific Context**: Marine heatwaves are extreme warming events that can have significant ecological and economic impacts. This analysis quantifies their changing characteristics.

### 4. `annual_psd.py`
**Purpose**: Calculate annual power spectral density (PSD) of daily SST variations

**Key Functions**:
- `ext_ldays(da)`: Removes leap days for consistent annual analysis
- `calc_ps(x)`: Computes power spectrum using FFT
- `calc_freq(x, dt=1)`: Calculates frequency array for power spectrum
- `annual_calc_ps(da, var='analysed_ssta')`: Annual PSD calculation wrapper

**Scientific Context**: Power spectral analysis reveals the frequency characteristics of SST variability, helping identify dominant timescales of ocean variability.

### 5. `thermal_memory_arctan.py`
**Purpose**: Alternative thermal memory calculation using arctangent regression model

**Key Functions**:
- `model_arctan(x, coeff)`: Arctangent model function for autocorrelation fitting
- `arctan_tau(year_idx)`: Annual thermal memory calculation using arctangent model

**Scientific Context**: The arctangent model provides an alternative approach to quantify thermal memory by fitting autocorrelation functions with a physically-motivated decay function.

## Data Requirements

The analysis functions require the following datasets:
- **OISST**: Daily sea surface temperature (1982-2022)
- **ERA5**: 2-meter air temperature and 10-meter wind speed
- **Mixed Layer Depth**: Climatological and trend data from Sallée et al. (2021)

## Constants and Parameters

Key physical constants used throughout the analysis:
- `sigma`: Stefan-Boltzmann coefficient (5.67×10⁻⁸ W m⁻²K⁻⁴)
- `rho_a`: Surface air density (1.22 kg m⁻³)
- `rho_w`: Seawater density (1.025×10³ kg m⁻³)
- `c_ap`: Specific heat of air (1.0005×10³ J kg⁻¹K⁻¹)
- `c_wp`: Specific heat of seawater (3.996×10³ J kg⁻¹K⁻¹)
- `L_w`: Latent heat of vaporization (2.5×10⁶ J kg⁻¹)

## Statistical Methods

- **Mann-Kendall Trend Test**: Non-parametric test for monotonic trends
- **Theil-Sen Slope Estimation**: Robust method for trend slope calculation
- **Autoregressive Analysis**: Time series modeling for memory timescale estimation

## Usage Notes

1. All functions are designed to work with xarray datasets for efficient array operations
2. Many functions use `xr.apply_ufunc` for vectorized operations across spatial dimensions
3. Parallel processing is implemented where computationally intensive (MHW analysis, arctangent fitting)
4. Missing data (NaN) is handled consistently across all functions

## References

- **Primary Paper**: Nature Climate Change article on declining thermal memory
- **Mixed Layer Depth**: Sallée, J.-B. et al. (2021). Nature 593, 51–56
- **Marine Heatwaves**: Oliver, E. C. J. et al. (2016). Geophysical Research Letters
- **Statistical Methods**: Hussain, M. M. et al. (2019) for Mann-Kendall implementation

## Citation
[https://www.nature.com/articles/s41558-025-02245-w](https://doi.org/10.1038/s41558-025-02245-w)
