from scipy import stats
import marineHeatWaves as mhw
import xarray as xr
import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymannkendall as mk


header = '/data2/users/lch204ok/persistence/41yr/'
dir_OISST = header + 'OISST/'

t_initial_vector = datetime.date(1982,1,1).toordinal()
t_final_veoctor =datetime.date(2021,12,31).toordinal()
t_vector = np.arange(t_initial_vector,t_final_veoctor+1)

sst_global = xr.open_dataset(dir_OISST+'1982-2022-1degree_SST.nc').sel(time=slice('1982-01-01','2021-12-31'))

#
# Mann-Kendall Trend test (modifie from Hussain et al.,(2019))
#
def trend_test(x):
    if np.isnan(np.sum(x)): trend, slope = np.nan, np.nan
    else:
        mk_x = mk.original_test(x,alpha=0.05)
        if mk_x.trend == 'increasing': trend = 1
        elif mk_x.trend == 'decreasing': trend = -1
        else: trend = 0
        #slope = mk_x.slope
    return trend#, slope
#
# End of trend_test
#

#
# Theil-Sen trend & marineheatwaves.py(modified from Oliver, et al., (2018); Oliver, et al., (2016))
#
def TS_for_MHWduration(sst):
    
    if np.sum(~np.isnan(sst))!=len(sst):
        signif, slope = np.nan, np.nan
    else:
        mhws, clim = mhw.detect(t_vector, sst)
        mhwBlock = mhw.blockAverage(t_vector, mhws)

        center_year = mhwBlock['years_centre']
        X = center_year-center_year.mean()

        y = mhwBlock['duration']
        valid = ~np.isnan(y) # non-NaN indices
        #
        #
        # Perform linear regression over valid indices
        if np.sum(~np.isnan(y)) == 0: # If at least one non-NaN value
            slope, signif = np.nan, np.nan
        else:
            slope, y0, beta_lr, beta_up = stats.mstats.theilslopes(y[valid], X[valid], alpha=1-0.05)
            signif = trend_test(y[valid])

    return signif, slope


def main(lat_idx):
    global dir_OISST
    global t_vector, sst_global

    sliced_data = sst_global.isel(lat=slice(lat_idx*10,(lat_idx+1)*10))
    sst_np = sliced_data['analysed_sst']
    lat_data = sliced_data['lat']
    lon_data = sliced_data['lon']

    duration_trend = np.zeros([2,sst_np.shape[1],sst_np.shape[2]])
    duration_trend = np.apply_along_axis(TS_for_MHWduration,0,sst_np)

    duration_trend_xr = xr.Dataset(
        data_vars=dict(
            significance = (['lat','lon'],duration_trend[0,:,:]),
            linear_slope = (['lat','lon'],duration_trend[1,:,:])
            ),
        coords=dict(
            lat = lat_data,
            lon = lon_data,
            )
        )
    name_tag_1 = '111111111222222222'
    name_tag_a = 'ABCDEFGHIABCDEFGHI'

    duration_trend_xr.to_netcdf(dir_OISST+'1982-2021_MHW_duration_MK_trend_test_'+name_tag_1[lat_idx]+name_tag_a[lat_idx]+'.nc')
    print(lat_idx,' is well done')

#
# execution part (uses 18 cores)
#
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(main, lat_idx) for lat_idx in range(18)]
    for future in as_completed(futures):
        future.result()
#
# end of execution part
#