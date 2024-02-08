'''
This code was written to calculate the two negative feedback rates (lambda_a and lambda_o).
It derives the average negative feedback rate and its changing trend from sea surface temperature, 10 meter wind speed, and 2 meter atmospheric temperature.
The bulk formulae used to calculate the latent and sensible heat fluxes are described in Large & Yeager, 2004.
'''

import xarray as xr
import numpy as np
from scipy import stats
import pymannkendall as mk

header = '/data2/users/lch204ok/persistence/'
dir_data = header+'41yr/data_availability/'
dir_OISST = header+'41yr/OISST/'
dir_ERA5 = header+'41yr/ERA5/'

'''
Essential coefficients and variables:
    sst_xr: sea surface temperature                   [K]
    t2m_xr: atmospheric temperature (2m)              [K]
    wind10_xr: wind speed (2m)                        [ms-1]
    mld_xr: mixed layer depth (climatology, trend)    [m; m year-1]
    tau: annual memory timescale of the upper ocean   [day]
    sigma: Stefan-Boltzmann coefficient               [Wm2K-4]
    rho_a: surface air density                        [kgm-3]
    rho_w: sea water density                          [kgm-3]
    c_ap: specific heat of the air                    [Jkg-1K-1]
    c_wp: specific heat of the sea water              [Jkg-1K-1]
    L_w: latent heat of vaporization                  [Jkg-1]
    day2sec: day to second                            [s]
    C_d: drag coefficient                             [dimensionless]
    C_e: transfer coefficient for evaporation         [dimensionless]
    C_h: transfer coefficient for sensible heat       [dimensionless]
'''
sst_xr = xr.open_dataset(dir_OISST+'1982-2022_analysed_sst.nc')
t2m_xr  = xr.open_dataset(dir_ERA5+'2mtemperature/1982-2022_t2m.nc')
wind10_xr = xr.open_dataset(dir_ERA5+'wind10/1982-2022_wind10.nc')
mld_xr = xr.open_dataset(header+'1970-2018_MLD_1_degree_Sallee.nc')
tau_xr = xr.open_dataarray(dir_data+'1982-2022_annual_thermal_memory.nc')
sigma = 5.67e-8
rho_a = 1.22
rho_w = 1.025e3
c_ap = 1.0005e3
c_wp = 3.996e3
day2sec = 60*60*24
L_w = 2.5*1e6
C_d = (2.7/wind10_xr['wind10'] + 0.142 + wind10_xr['wind10']/13.09)*1e-3 
C_e = 34.6*np.sqrt(C_d)*1e-3 

def stability(t2m,sst):
    gamma = -0.98*1e-3 # dry adiabatic rapse rate
    func = lambda x, y: x+gamma*(-2)-y
    return xr.apply_ufunc(func, t2m, sst)
zeta = stability(t2m_xr['t2m'],sst_xr['analysed_sst'])

def vfunc_Ch(C_d, zeta):
    func = lambda x, y: xr.where(y>0, 18*np.sqrt(x)*1e-3, 32.7*np.sqrt(x)*1e-3)
    return xr.apply_ufunc(func, C_d, zeta)
C_h = vfunc_Ch(C_d, zeta) # transfer coefficient for sensible heat

'''
Negative feedback rate back toward atm(lambda_a) [Wm-2K-1]
    lambda_lw: radiative heat (longwave)
    lambda_sh: sensible heat
    lambda_lh: latent heat
'''
lambda_lw = 4*sigma*t2m_xr['t2m']**3

def calc_sh(C_h,wind10):
    func = lambda x, y: rho_a*c_ap*x*y
    return xr.apply_ufunc(func, C_h, wind10)
lambda_sh = calc_sh(C_h,wind10_xr['wind10'])
    
def calc_lh(C_e,wind10,t2m):
    q1 = 0.98*640380 # [kgm-3]
    q2 = -5107.4     # [K]
    func = lambda x, y, z: -L_w*x*y*(q1*q2/z**2)*np.exp(q2/z)
    return xr.apply_ufunc(func,C_e,wind10,t2m)
lambda_lh = calc_lh(C_e,wind10_xr['wind10'],t2m_xr['t2m'])

heatflux_feedback = lambda_lw.to_dataset(name = 'lambda_lw')
heatflux_feedback['lambda_sh'] = lambda_sh
heatflux_feedback['lambda_lh'] = lambda_lh
del lambda_lw, lambda_sh, lambda_lh
heatflux_feedback = heatflux_feedback.assign_attrs(
    units="Wm-2K-1",
    discription="daily negative heatflux feedback rate",
    lambda_lw="via longwave radiation",
    lambda_sh="via sensible heatflux",
    lambda_lh="via latent heatflux",
    start_time ="1982-01-01",
    end_time = "2022-01-01"
)
heatflux_feedback.to_netcdf(dir_data+'1982-2022_daily_heatflux_feedback.nc')

'''
Negative feedback rate via oceanic processes (lambda_o) [Wm-2K-1]
    lambda_a: oceanc processes (oceanic)
'''
tau_sec = tau_xr*day2sec
heat_capa_w = mld_xr['climato_mld']*c_wp*rho_w  # climatology heat capacity of the upper ocean
annual_hf_fb_total = 1/tau_sec*heat_capa_w
annual_hf_fb_total = annual_hf_fb_total.transpose('time','lat','lon')
annual_hf_fb_a = heatflux_feedback.groupby('time.year').mean().rename({'year':'time'})
annual_hf_fb_o = annual_hf_fb_total - (annual_hf_fb_a['lambda_lw']+annual_hf_fb_a['lambda_sh']+annual_hf_fb_a['lambda_lh']).data

annual_heatflux_feedback = annual_hf_fb_a['lambda_lw'].to_dataset(name='lambda_lw')
annual_heatflux_feedback['lambda_sh'] = annual_hf_fb_a['lambda_sh']
annual_heatflux_feedback['lambda_lh'] = annual_hf_fb_a['lambda_lh']
annual_heatflux_feedback['lambda_o'] = annual_hf_fb_o

annual_heatflux_feedback = xr.Dataset(
    data_vars=dict(
        lambda_lw=(['time','lat','lon'],annual_hf_fb_a['lambda_lw'].data),
        lambda_sh=(['time','lat','lon'],annual_hf_fb_a['lambda_sh'].data),
        lambda_lh=(['time','lat','lon'],annual_hf_fb_a['lambda_lh'].data),
        lambda_o=(['time','lat','lon'],annual_hf_fb_o.data)
    ),
    coords=dict(
        time=annual_hf_fb_o['time'].data,
        lat=annual_hf_fb_o['lat'].data,
        lon=annual_hf_fb_o['lon'].data
    ),
    attrs=dict(
        units="Wm-2K-1",
        discription='annual negative heatflux feedback rate',
        lambda_lw="via longwave radiation",
        lambda_sh="via sensible heatflux",
        lambda_lh="via latent heatflux",
        lambda_o='via oceanic process',
        start_time='year of 1982',
        end_time='year of 2022'
    )
)
annual_heatflux_feedback.to_netcdf(dir_data+'1982-2022_annual_heatflux_feedback.nc')

'''
Trend of the heatflux feedback rate [Wm-2K-1 year-1]
    lambda_a: toward atmosphere
    lambda_o: oceanic processes
'''
#
# 1982-2021
#
x = np.arange(1982,2022)

def trend_test(x):
    if np.isnan(np.sum(x)): trend, slope = np.nan, np.nan
    else:
        mk_x = mk.original_test(x,alpha=0.05)
        if mk_x.trend == 'increasing': trend = 1
        elif mk_x.trend == 'decreasing': trend = -1
        else: trend = 0
        #slope = mk_x.slope
    return trend#, slope

def Theilsen_test(y):
    if np.sum(~np.isnan(y))!=len(y):
        signif, slope = np.nan, np.nan
    else:
        slope, y0, beta_lr, beta_up = stats.mstats.theilslopes(y,x, alpha=1-0.05)
        signif = trend_test(y)

    return signif, slope

annual_lambda_a_2021 = annual_heatflux_feedback.sel(time=slice('1982-01-01','2021-12-31'))['lambda_lw']+\
                annual_heatflux_feedback.sel(time=slice('1982-01-01','2021-12-31'))['lambda_sh']+\
                annual_heatflux_feedback.sel(time=slice('1982-01-01','2021-12-31'))['lambda_lh']

annual_lambda_a_2021 = xr.where(np.isnan(tau_xr.sel(time=slice('1982-01-01','2021-12-31'))),np.nan*annual_lambda_a_2021, annual_lambda_a_2021)
annual_lambda_o_2021 = annual_heatflux_feedback.sel(time=slice('1982-01-01','2021-12-31'))['lambda_o']

trend_lambda_a_2021 = np.apply_along_axis(Theilsen_test,0,annual_lambda_a_2021.data[:40,:,:])
trend_lambda_o_2021 = np.apply_along_axis(Theilsen_test,0,annual_lambda_o_2021.data[:40,:,:])

trend_heatflux_feedback_2021 = xr.Dataset(
    data_vars = dict(
        lambda_a = (['types','lat','lon'],trend_lambda_a_2021),
        lambda_o = (['types','lat','lon'],trend_lambda_o_2021)
    ),
    coords=dict(
        types = ['trend','linear slope'],
        lat = annual_heatflux_feedback['lat'].data,
        lon = annual_heatflux_feedback['lon'].data
    ),

    attrs = dict(
        units = "Wm-2K-2 year-1",
        discription = "Changes in heatflux feedback rate",
        lambda_a = 'toward atmosphere',
        lambda_o = 'via oceanic processes',
        range = '1982-2021'
    )
)

trend_heatflux_feedback_2021.to_netcdf(dir_data+'1982-2021_trend_of_heatflux_feedback.nc')

#
# 1982-2022
#
x = np.arange(1982,2023)

annual_lambda_a_2022 = annual_heatflux_feedback.sel(time=slice('1982-01-01','2022-12-31'))['lambda_lw']+\
                annual_heatflux_feedback.sel(time=slice('1982-01-01','2022-12-31'))['lambda_sh']+\
                annual_heatflux_feedback.sel(time=slice('1982-01-01','2022-12-31'))['lambda_lh']

annual_lambda_a_2022 = xr.where(np.isnan(tau_xr),np.nan*annual_lambda_a_2022, annual_lambda_a_2022)
annual_lambda_o_2022 = annual_heatflux_feedback.sel(time=slice('1982-01-01','2022-12-31'))['lambda_o']

trend_lambda_a = np.apply_along_axis(Theilsen_test,0,annual_lambda_a_2022.data[:41,:,:])
trend_lambda_o = np.apply_along_axis(Theilsen_test,0,annual_lambda_o_2022.data[:41,:,:])

trend_heatflux_feedback_2022 = xr.Dataset(
    data_vars = dict(
        lambda_a = (['types','lat','lon'],trend_lambda_a),
        lambda_o = (['types','lat','lon'],trend_lambda_o)
    ),
    coords=dict(
        types = ['trend','linear slope'],
        lat = annual_heatflux_feedback['lat'].data,
        lon = annual_heatflux_feedback['lon'].data
    ),

    attrs = dict(
        units = "Wm-2K-2 year-1",
        discription = "Changes in heatflux feedback rate",
        lambda_a = 'toward atmosphere',
        lambda_o = 'via oceanic processes',
        range = '1982-2022'
    )
)

trend_heatflux_feedback_2022.to_netcdf(dir_data+'1982-2022_trend_of_heatflux_feedback.nc')
