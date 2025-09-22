'''
Jun. 8th, 2023 Chaehyoeng Lee.

This code was written to quantitatively calculate the annual variation in the power spectral density of daily mean sea surface temperature.
''' 

import xarray as xr
import numpy as np
import pandas as pd

header = '/data2/users/lch204ok/'
dir_data = header+'persistence/data_availability/'

sst_ncei = xr.open_dataset(dir_data+'1982-2022_analysed_sst.nc')

def ext_ldays(da):
    """
    Remove leap days (February 29th) for consistent annual analysis.
    
    Args:
        da: xarray dataset with time dimension
    
    Returns:
        da_noleap: dataset with leap days removed
    """
    time_index = pd.to_datetime(da['time'].values) 
    leap_day_mask = (time_index.month == 2) & (time_index.day == 29) # Identify leap days
    unique_time_index = time_index[~leap_day_mask] # Exclude leap days
    non_leap_day_mask = da['time'].isin(unique_time_index) # Create mask for non-leap days
    da_noleap = da.sel(time=non_leap_day_mask) # Select non-leap days only
    return da_noleap

def calc_ps(x):
    """
    Calculate power spectral density using FFT.
    
    Args:
        x: input time series
    
    Returns:
        ps: normalized power spectrum (positive frequencies only)
    """
    nt = len(x)
    npositive = nt//2
    pslice = slice(1, npositive) # take only positive frequencies (excluding DC component)
    fft_result = np.fft.fft(x)[pslice] # Compute FFT
    ps = np.abs(fft_result) ** 2  # Power spectrum
    ps *= 2         # Double to account for energy in negative frequencies
    ps /= nt**2     # Normalization for proper power spectrum
    return ps

def calc_freq(x, dt=1):
    """
    Calculate frequency array for power spectrum.
    
    Args:
        x: input time series
        dt: time step (default=1 day)
    
    Returns:
        freq: frequency array (positive frequencies only)
    """
    nt = len(x)
    npositive = nt//2
    pslice = slice(1, npositive) # take only positive frequencies (excluding DC)
    freq = np.fft.fftfreq(nt, d=1/dt)[pslice]  # Generate frequency array
    return freq
    
def annual_calc_ps(da, var='analysed_ssta'):
    """
    Calculate annual power spectral density for each year.
    
    Args:
        da: xarray dataset containing SST anomaly data
        var: variable name to process (default: 'analysed_ssta')
    
    Returns:
        annual_ps: annual power spectra with frequency coordinates
    """
    da = ext_ldays(da[var])  # Remove leap days for consistent 365-day years
    # Group dataset by year and apply power spectrum calculation
    annual_da = da.groupby('time.year')
    annual_ps = annual_da.apply(
        lambda x: xr.apply_ufunc(
            lambda y: np.apply_along_axis(calc_ps, 0, y),
            x,
            dask='parallelized',
            input_core_dims=[['time']],  # Time is the core dimension for input
            output_core_dims=[['frequency']],  # Frequency is the core dimension for output
            output_dtypes=[float],
            vectorize=True
        )
    )
    # Add frequency coordinates based on 365-day year
    annual_ps = annual_ps.assign_coords({'frequency':('frequency',calc_freq(da.sel({'lat':0,'lon':0},method='nearest').isel(time=slice(0,365))))})
    return annual_ps
#
# Cacluation 
#
annual_ps = annual_calc_ps(sst_ncei)
