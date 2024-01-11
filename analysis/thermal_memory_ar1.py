'''
Jan. 10th, 2023 Chaehyeong Lee

This library was written to calculate surface theraml memory from sea surface temperature anmoalies by using Autoregressive-1 model.
It also includes functions for quantitatively deriving changes in time-series data using the Mann-Kendall trend test and the Theil-sen trend test.
'''
import numpy as np
import pymannkendall as mk
from numba import jit

'''
Input   detrend_ssta: one dimensional detrended SST anomalies
'''
@jit(nopython=True)
def autocorr_lag1(x): # lag-1 autocorrelation
    if np.isnan(np.sum(x)): result = np.nan 
    else: result = np.corrcoef(x[:-1], x[1:])[0,1]
    return result


def add_lag0(lag_1): # add lag-0 autocorrelation(1) in front of lag-1 autocorrelation.
    len_lat = lag_1.shape[0]
    len_lon = lag_1.shape[1]
    result = np.ones([2,len_lat,len_lon])
    result[1,:,:] = lag_1
    return result

def model_exp(x,coeff): # simple exponentially decaying function
    return np.exp(-coeff*x)


def memory_timescale(autocorr_coeff): 
    if np.isnan(np.sum(autocorr_coeff)): persist, half_persist = np.nan, np.nan
    else: 
        lags=np.array([0,1])
        persist = -1/np.log(autocorr_coeff)
        half_persist = -(1-np.log(2))/np.log(autocorr_coeff)
    return persist, half_persist

#
# Mann-Kendall Trend test (modifie from Hussain et al.,(2019))
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
