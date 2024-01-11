'''
This library was written to calculate surface theraml memory from sea surface temperature anmoalies 
by using the arctangent regressvie with autocorrelation cofficient.
'''
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from statsmodels.tsa.stattools import acf

f_dir = '/data2/users/lch204ok/persistence/41yr/OISST/'
ssta = xr.open_dataset(f_dir+'1982-2022-1degree_deseason_detrend-SST.nc')['analysed_sst']

def model_arctan(x,coeff): 
    return 1-0.5*np.arctan(coeff*x)

def arctan_tau(year_idx):
    global ssta, f_dir

    date_init = date(year_idx,1,1).strftime('%Y-%m-%d')
    date_last = date(year_idx,12,31).strftime('%Y-%m-%d')

    ssta_target = ssta.sel(time=slice(date_init,date_last)).data
    target_acf_30 = np.zeros([30,180,360])*np.NaN
    target_acf_20 = np.zeros([20,180,360])*np.NaN
    target_acf_10 = np.zeros([10,180,360])*np.NaN
    target_acf_5  = np.zeros([5,180,360])*np.NaN
    target_acf_2  = np.zeros([2,180,360])*np.NaN

    tau = np.zeros([5,180,360])*np.NaN

    lag_data = np.array([2,5,10,20,30])
    fit_lags = np.arange(30)

    for lat in range(180):
        for lon in range(360):
            print(lat)
            if np.isnan(np.sum(ssta_target[:,lat,lon])) == False:
                target_acf_30[:,lat,lon] = acf(ssta_target[:,lat,lon],nlags=29)
                target_acf_20[:,lat,lon] = acf(ssta_target[:,lat,lon],nlags=19)
                target_acf_10[:,lat,lon] = acf(ssta_target[:,lat,lon],nlags=9)
                target_acf_5[:,lat,lon]  = acf(ssta_target[:,lat,lon],nlags=4)
                target_acf_2[:,lat,lon]  = acf(ssta_target[:,lat,lon],nlags=1)

                popt_30, _ = curve_fit(model_arctan,fit_lags,target_acf_30[:,lat,lon])
                popt_20, _ = curve_fit(model_arctan,fit_lags[:20],target_acf_20[:,lat,lon])
                popt_10, _ = curve_fit(model_arctan,fit_lags[:10],target_acf_10[:,lat,lon])
                popt_5, _ = curve_fit(model_arctan,fit_lags[:5],target_acf_5[:,lat,lon])
                popt_2, _ = curve_fit(model_arctan,fit_lags[:2],target_acf_2[:,lat,lon])

                for lag_idx, target_lag in enumerate(lag_data):
                    tau[lag_idx,lat,lon] = np.tan(2-2/np.e)/locals()['popt_'+str(target_lag)][0]

    
    time_data = np.array([date_init])
    lat_data = np.arange(-89.5,90,1)
    lon_data = np.arange(.5,360,1)
    tau_xr = xr.Dataset(data_vars=dict(
        thermal_memory = (['lag','time','lat','lon'],tau)
        ),
        coords={'lag':lat_data,'time':time_data,'lat':lat_data,'lon':lon_data},

        attrs=dict(
            description='Surface Thermal Memory with respect to Arctangent model',
            unit='Days'
        )
    )

    tau_xr['lon'].attrs.update({'standard_name': 'longitude', 'long_name': 'Longitude', 'units': 'degrees_east', '_CoordinateAxisType': 'Lon'})
    tau_xr['lat'].attrs.update({'standard_name': 'latitude', 'long_name': 'Latitude', 'units': 'degrees_north', '_CoordinateAxisType': 'Lat'})
    
    tau_xr.to_netcdf(f_dir+'Arctan/'+date_init[:4]+'_tau.nc')
    
#############################################################################################################
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(arctan_tau, year_idx) for year_idx in range(1982,2023)]
    for future in as_completed(futures):
        future.result()
