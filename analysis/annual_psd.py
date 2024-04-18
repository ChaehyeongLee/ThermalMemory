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
    time_index = pd.to_datetime(da['time'].values) 
    leap_day_mask = (time_index.month == 2) & (time_index.day == 29) # Identify leap days (February 29th)
    unique_time_index = time_index[~leap_day_mask] # Ensure we only have unique values by excluding leap days
    non_leap_day_mask = da['time'].isin(unique_time_index) # Use isin to create a mask for the times that are not leap days
    da_noleap = da.sel(time=non_leap_day_mask) # Finally, select the non-leap days in the Dataset
    return da_noleap

def calc_ps(x):
    nt = len(x)
    npositive = nt//2
    pslice = slice(1, npositive) # take only the positive frequencies (w/o '0')
    fft_result = np.fft.fft(x)[pslice] # Compute the power spectrum
    ps = np.abs(fft_result) ** 2 /nt
    ps *= 2         # Double to account for the energy in the negative frequencies
    ps /= nt**2     # Normalizeation for Power Spectrum
    return ps

def calc_freq(x,dt=1):
    nt = len(x)
    npositive = nt//2
    pslice = slice(1, npositive) # take only the positive frequencies (w/o '0')
    freq = np.fft.fftfreq(nt,d=1/dt)[pslice]
    return freq
    
# Function to apply the power spectrum calculation across the dataset for each year
def annual_calc_ps(da,var='analysed_ssta'):
    da = ext_ldays(da[var])
    # Group the dataset by year and then apply the power spectrum calculation to each group
    annual_da = da.groupby('time.year')
    annual_ps = annual_da.apply(
        lambda x: xr.apply_ufunc(
            lambda y: np.apply_along_axis(calc_ps, 0, y),
            x,
            dask='parallelized',
            input_core_dims=[['time']],  # Specify 'time' as the core dimension
            output_core_dims=[['frequency']],  # Specify 'frequency' as the core dimension in the output
            output_dtypes=[float],
            vectorize=True
        )
    )
    annual_ps = annual_ps.assign_coords({'frequency':('frequency',calc_freq(da.sel({'lat':0,'lon':0},method='nearest').isel(time=slice(0,365))))})
    return annual_ps
#
# Cacluation 
#
annual_ps = annual_calc_ps(sst_ncei)
