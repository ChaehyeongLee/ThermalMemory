'''
Jan. 10th, 2023 Chaehyeong Lee

This library was written to calculate surface theraml memory from sea surface temperature anmoalies by using Autoregressive-1 model.
It also includes functions for quantitatively deriving changes in time-series data using the Mann-Kendall trend test and the Theil-sen trend test.
'''
import numpy as np
import xarray as xr
import pymannkendall as mk
from numba import jit

'''
Input   Sea surface temperature time series data (not less than 1 dimension)

'''
@jit(nopython=True)
def memory_timescale(da,vars, option='annual'):
    da = da[vars]
    tau_func = lambda x: -1/np.log(np.corrcoef(x[:-1],x[1:])[0,1])

    if option=='annual': 
        annual_da = da.groupby('time.year')
        tau = annual_da.apply(
            lambda y: xr.apply_ufunc(
                lambda z: np.apply_along_axis(tau_func, 0, z),
                y,
                dask='parallelized',
                input_core_dims=[['time']], 
                output_core_dims=[[]],  
                output_dtypes=[float],
                vectorize=True
            )
        )
    else: 
        tau = xr.apply_ufunc(
            lambda y: np.apply_along_axis(tau_func,0,y),
            da,
            dask='parallelized',
            input_core_dims=[['time']],
            output_core_dims=[[]],
            output_dtypes=[float],
            vectorize=True
        )
    
    return tau

#
# Mann-Kendall Trend test (modifie from Hussain et al.,(2019))
# slope follows Theil-Sen method
#
def trend_test(x):
    if np.isnan(np.sum(x)): trend, slope = np.nan, np.nan
    else:
        mk_x = mk.original_test(x)
        if mk_x.trend == 'increasing': trend = 1
        elif mk_x.trend == 'decreasing': trend = -1
        else: trend = 0
        slope = mk_x.slope
    return trend, slope
