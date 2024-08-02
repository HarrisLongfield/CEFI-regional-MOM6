##########################################################################
#IMPORT PACKAGES
##########################################################################

import cartopy.feature as feature
import cartopy.crs as ccrs
from calendar import month_abbr
import matplotlib.dates as mdates
from pathlib import Path 
import subprocess 
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import datetime
import numpy as np
import pandas as pd
import xarray
import xesmf
import os 
from glob import glob
from plot_common import autoextend_colorbar, corners, get_map_norm, open_var, add_ticks, annotate_skill, save_figure
from scipy.stats import ttest_1samp
import statistics
import scipy 

##########################################################################
#COMPUTE THE GULFSTREAM INDEX
##########################################################################

def compute_gs(ssh, data_grid=None):
    lons = np.arange(360-72, 360-51.9, 1)
    lats = np.arange(36, 42, 0.1)
    target_grid = {'lat': lats, 'lon': lons}

    if data_grid is None:
        data_grid = {'lat': ssh.lat, 'lon': ssh.lon}

    ssh_to_grid = xesmf.Regridder(
        data_grid,
        target_grid,
        method='bilinear'
    )

    # Interpolate the SSH data onto the index grid.
    regridded = ssh_to_grid(ssh)

    # Find anomalies relative to the calendar month mean SSH over the full model run.
    anom = regridded.groupby('time.month') - regridded.groupby('time.month').mean('time')

    # For each longitude point, the Gulf Stream is located at the latitude with the maximum SSH anomaly variance.
    stdev = anom.std('time')
    amax = stdev.argmax('lat').compute()
    gs_points = stdev.lat.isel(lat=amax).compute()

    # The index is the mean latitude of the Gulf Stream, divided by the standard deviation of the mean latitude of the Gulf Stream.
    index = ((anom.isel(lat=amax).mean('lon')) / anom.isel(lat=amax).mean('lon').std('time')).compute()

    # Move times to the beginning of the month to match observations.
    monthly_index = index.to_pandas().resample('1MS').first()
    return monthly_index, gs_points

##########################################################################
#FUNCTION TO OPEN VARIABLES
##########################################################################

def open_var(pp_root, kind, var):
    freq = 'daily' if 'daily' in kind else 'monthly'
    longslice = '19930101-19981231' if freq == 'daily' else '199301-199812'
    longfile = os.path.join(pp_root, 'pp', kind, 'ts', freq, '5yr', f'{kind}.{longslice}.{var}.nc')
    if os.path.isfile(longfile):
        os.system(f'dmget {longfile}')
        return xarray.open_dataset(longfile)[var]
    elif len(glob(os.path.join(pp_root, 'pp', kind, 'ts', freq, '1yr', f'{kind}.*.{var}.nc'))) > 0:
        files = glob(os.path.join(pp_root, 'pp', kind, 'ts', freq, '1yr', f'{kind}.*.{var}.nc'))
        os.system(f'dmget {" ".join(files)}')
        return xarray.open_mfdataset(files)[var]
    elif len(glob(os.path.join(pp_root, 'pp', kind, 'ts', freq, '5yr', f'{kind}.*.{var}.nc'))) > 0:
        files = glob(os.path.join(pp_root, 'pp', kind, 'ts', freq, '5yr', f'{kind}.*.{var}.nc'))
        os.system(f'dmget {" ".join(files)}')
        return xarray.open_mfdataset(files)[var]
    else:
        raise Exception('Did not find postprocessed files')

##########################################################################
#PREPARE OBSERVATIONAL DATA
##########################################################################

obs_sst = xarray.open_mfdataset('/work/vnk/obs_and_reanalyses/oisst_v2/sst.mon.mean.nc')['sst']
obs_ssh_points = xarray.open_mfdataset('/home/Andrew.C.Ross/git/nwa12/experiments/COBALT_2023_04/data/satellite_ssh_points.nc') 
obs_ssh_indexes = pd.read_pickle('/home/Andrew.C.Ross/git/nwa12/experiments/COBALT_2023_04/data/satellite_ssh_index.pkl').resample('1YS').mean()

model_grid = xarray.open_dataset('../data/geography/ocean_static.nc')
obs_sst_grid = xarray.open_mfdataset('/work/vnk/obs_and_reanalyses/oisst_v2/sst.mon.mean.grid.nc')  

obsregrd = xesmf.Regridder(obs_sst_grid, {'lat': model_grid.geolat, 'lon': model_grid.geolon}, 
        method='bilinear', 
        unmapped_to_nan=True)

obs_regriddeda = obsregrd(obs_sst)
obs_regridded = obs_regriddeda.sel(time=slice('1993','2007'))

##########################################################################
#PATHS TO FILES
##########################################################################

rootmed1 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_1/gfdl.ncrc5-intel22-prod') 
rootlow1 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_1/gfdl.ncrc5-intel22-prod')
rootmed2 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_2/gfdl.ncrc5-intel22-prod') 
rootlow2 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_2/gfdl.ncrc5-intel22-prod')
rootmed3 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_3/gfdl.ncrc5-intel22-prod') 
rootlow3 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_3/gfdl.ncrc5-intel22-prod')
rootmed4 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_4/gfdl.ncrc5-intel22-prod') 
rootlow4 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_4/gfdl.ncrc5-intel22-prod')
rootmed5 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_5/gfdl.ncrc5-intel22-prod') 
rootlow5 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_5/gfdl.ncrc5-intel22-prod')

##########################################################################
#GRAB VARS 
##########################################################################

#SST
sstwholemed1 = open_var(rootmed1, 'ocean_monthly', 'tos').resample(time = "1MS").first() 
sstwholemed2 = open_var(rootmed2, 'ocean_monthly', 'tos').resample(time = "1MS").first() 
sstwholemed3 = open_var(rootmed3, 'ocean_monthly', 'tos').resample(time = "1MS").first()
sstwholemed4 = open_var(rootmed4, 'ocean_monthly', 'tos').resample(time = "1MS").first() 
sstwholemed5 = open_var(rootmed5, 'ocean_monthly', 'tos').resample(time = "1MS").first() 

sstwholelow1 = open_var(rootlow1, 'ocean_monthly', 'tos').resample(time = "1MS").first() 
sstwholelow2 = open_var(rootlow2, 'ocean_monthly', 'tos').resample(time = "1MS").first() 
sstwholelow3 = open_var(rootlow3, 'ocean_monthly', 'tos').resample(time = "1MS").first()
sstwholelow4 = open_var(rootlow4, 'ocean_monthly', 'tos').resample(time = "1MS").first()
sstwholelow5 = open_var(rootlow5, 'ocean_monthly', 'tos').resample(time = "1MS").first()

#SSH 
sshwholemed1 = open_var(rootmed1, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 
sshwholemed2 = open_var(rootmed2, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 
sshwholemed3 = open_var(rootmed3, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 
sshwholemed4 = open_var(rootmed4, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 
sshwholemed5 = open_var(rootmed5, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 

sshwholelow1 = open_var(rootlow1, 'ocean_monthly', 'ssh').resample(time = "1MS").first() 
sshwholelow2 = open_var(rootlow2, 'ocean_monthly', 'ssh').resample(time = "1MS").first()
sshwholelow3 = open_var(rootlow3, 'ocean_monthly', 'ssh').resample(time = "1MS").first()
sshwholelow4 = open_var(rootlow4, 'ocean_monthly', 'ssh').resample(time = "1MS").first()
sshwholelow5 = open_var(rootlow5, 'ocean_monthly', 'ssh').resample(time = "1MS").first()

##########################################################################
#CALCULATE GSI INDEX
##########################################################################

modelgsiindexmed1, modelgsipointsmed1 = compute_gs(sshwholemed1, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexmed2, modelgsipointsmed2 = compute_gs(sshwholemed2, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexmed3, modelgsipointsmed3 = compute_gs(sshwholemed3, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexmed4, modelgsipointsmed4 = compute_gs(sshwholemed4, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexmed5, modelgsipointsmed5 = compute_gs(sshwholemed5, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))

modelgsiindexlow1, modelgsipointslow1 = compute_gs(sshwholelow1, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexlow2, modelgsipointslow2 = compute_gs(sshwholelow2, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexlow3, modelgsipointslow3 = compute_gs(sshwholelow3, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexlow4, modelgsipointslow4 = compute_gs(sshwholelow4, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
modelgsiindexlow5, modelgsipointslow5 = compute_gs(sshwholelow5, data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))

#ANNUALIZE

annual_gsi_index_med1 = modelgsiindexmed1.resample('1YS').mean()
annual_gsi_index_med2 = modelgsiindexmed2.resample('1YS').mean()
annual_gsi_index_med3 = modelgsiindexmed3.resample('1YS').mean()
annual_gsi_index_med4 = modelgsiindexmed4.resample('1YS').mean()
annual_gsi_index_med5 = modelgsiindexmed5.resample('1YS').mean()

annual_gsi_index_low1 = modelgsiindexlow1.resample('1YS').mean()
annual_gsi_index_low2 = modelgsiindexlow2.resample('1YS').mean()
annual_gsi_index_low3 = modelgsiindexlow3.resample('1YS').mean()
annual_gsi_index_low4 = modelgsiindexlow4.resample('1YS').mean()
annual_gsi_index_low5 = modelgsiindexlow5.resample('1YS').mean()

##########################################################################
#DEFINE MASKS AND COMPUTE AVERAGE ANNUAL SST 
##########################################################################

masks = xarray.open_dataset('/home/Andrew.C.Ross/git/nwa12/data/geography/masks/regions_30m.nc') 

weights_gom = masks['areacello'].where(masks['GOM']).fillna(0) 
weights_neus = masks['areacello'].where(masks['NEUS_LME']).fillna(0)

#FIND AVERAGE OBS SST OVER REGIONS
mean_sst_neus_obs = obs_regridded.weighted(weights_neus).mean(['yh', 'xh'])
mean_sst_gom_obs = obs_regridded.weighted(weights_gom).mean(['yh', 'xh']) 

#FIND AVERAGE MODEL SST OVER NEUS
mean_sst_neus_med1 = sstwholemed1.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_med2 = sstwholemed2.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_med3 = sstwholemed3.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_med4 = sstwholemed4.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_med5 = sstwholemed5.weighted(weights_neus).mean(['yh', 'xh']).compute() 

mean_sst_neus_low1 = sstwholelow1.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_low2 = sstwholelow2.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_low3 = sstwholelow3.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_low4 = sstwholelow4.weighted(weights_neus).mean(['yh', 'xh']).compute() 
mean_sst_neus_low5 = sstwholelow5.weighted(weights_neus).mean(['yh', 'xh']).compute() 

#FIND AVERAGE MODEL SST OVER GOM 
mean_sst_gom_med1 = sstwholemed1.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_med2 = sstwholemed2.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_med3 = sstwholemed3.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_med4 = sstwholemed4.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_med5 = sstwholemed5.weighted(weights_gom).mean(['yh', 'xh']).compute() 

mean_sst_gom_low1 = sstwholelow1.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_low2 = sstwholelow2.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_low3 = sstwholelow3.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_low4 = sstwholelow4.weighted(weights_gom).mean(['yh', 'xh']).compute() 
mean_sst_gom_low5 = sstwholelow5.weighted(weights_gom).mean(['yh', 'xh']).compute() 

#ANNUALIZE AVERAGE OBS SST OVER REGIONS
annual_neus_obs = mean_sst_neus_obs.resample(time='1YS').mean() 
annual_gom_obs = mean_sst_gom_obs.resample(time='1YS').mean() 

#ANNUALIZE MODEL SST OVER NEUS
annual_neus_med1 = mean_sst_neus_med1.resample(time='1YS').mean()
annual_neus_med2 = mean_sst_neus_med2.resample(time='1YS').mean()
annual_neus_med3 = mean_sst_neus_med3.resample(time='1YS').mean()
annual_neus_med4 = mean_sst_neus_med4.resample(time='1YS').mean()
annual_neus_med5 = mean_sst_neus_med5.resample(time='1YS').mean()

annual_neus_low1 = mean_sst_neus_low1.resample(time='1YS').mean()
annual_neus_low2 = mean_sst_neus_low2.resample(time='1YS').mean()
annual_neus_low3 = mean_sst_neus_low3.resample(time='1YS').mean()
annual_neus_low4 = mean_sst_neus_low4.resample(time='1YS').mean()
annual_neus_low5 = mean_sst_neus_low5.resample(time='1YS').mean()

#ANNUALIZE MODEL SST OVER GOM
annual_gom_med1 = mean_sst_gom_med1.resample(time='1YS').mean()
annual_gom_med2 = mean_sst_gom_med2.resample(time='1YS').mean()
annual_gom_med3 = mean_sst_gom_med3.resample(time='1YS').mean()
annual_gom_med4 = mean_sst_gom_med4.resample(time='1YS').mean()
annual_gom_med5 = mean_sst_gom_med5.resample(time='1YS').mean()

annual_gom_low1 = mean_sst_gom_low1.resample(time='1YS').mean()
annual_gom_low2 = mean_sst_gom_low2.resample(time='1YS').mean()
annual_gom_low3 = mean_sst_gom_low3.resample(time='1YS').mean()
annual_gom_low4 = mean_sst_gom_low4.resample(time='1YS').mean()
annual_gom_low5 = mean_sst_gom_low5.resample(time='1YS').mean()

##########################################################################
#CALCULATE CORRELATION COEFFICIENTS 
##########################################################################

#OBS COEFFICIENTS 
annual_neus_obs_corr = xarray.corr(annual_neus_obs.sel(time=slice('1995', '2007')).load(), pd.DataFrame.to_xarray(obs_ssh_indexes).sel(time=slice('1995', '2007')), dim='time').item()
annual_gom_obs_corr = xarray.corr(annual_gom_obs.sel(time=slice('1995', '2007')).load(), pd.DataFrame.to_xarray(obs_ssh_indexes).sel(time=slice('1995', '2007')), dim='time').item()

#NEUS COEFFICIENTS 
annual_neus_med1_corr = xarray.corr(annual_neus_med1.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med1), dim='time').item() 
annual_neus_med2_corr = xarray.corr(annual_neus_med2.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med2), dim='time').item() 
annual_neus_med3_corr = xarray.corr(annual_neus_med3.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med3), dim='time').item() 
annual_neus_med4_corr = xarray.corr(annual_neus_med4.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med4), dim='time').item() 
annual_neus_med5_corr = xarray.corr(annual_neus_med5.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med5), dim='time').item() 

annual_neus_low1_corr = xarray.corr(annual_neus_low1.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low1), dim='time').item() 
annual_neus_low2_corr = xarray.corr(annual_neus_low2.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low2), dim='time').item() 
annual_neus_low3_corr = xarray.corr(annual_neus_low3.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low3), dim='time').item() 
annual_neus_low4_corr = xarray.corr(annual_neus_low4.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low4), dim='time').item() 
annual_neus_low5_corr = xarray.corr(annual_neus_low5.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low5), dim='time').item() 

#GOM COEFFICIENTS
annual_gom_med1_corr = xarray.corr(annual_gom_med1.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med1), dim='time').item() 
annual_gom_med2_corr = xarray.corr(annual_gom_med2.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med2), dim='time').item() 
annual_gom_med3_corr = xarray.corr(annual_gom_med3.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med3), dim='time').item() 
annual_gom_med4_corr = xarray.corr(annual_gom_med4.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med4), dim='time').item() 
annual_gom_med5_corr = xarray.corr(annual_gom_med5.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_med5), dim='time').item() 

annual_gom_low1_corr = xarray.corr(annual_gom_low1.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low1), dim='time').item() 
annual_gom_low2_corr = xarray.corr(annual_gom_low2.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low2), dim='time').item() 
annual_gom_low3_corr = xarray.corr(annual_gom_low3.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low3), dim='time').item() 
annual_gom_low4_corr = xarray.corr(annual_gom_low4.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low4), dim='time').item() 
annual_gom_low5_corr = xarray.corr(annual_gom_low5.sel(time=slice('1995','2007')), pd.DataFrame.to_xarray(annual_gsi_index_low5), dim='time').item() 

##########################################################################
#EVALUATE STATISTICS
##########################################################################

neuslowlist = [annual_neus_low1_corr, annual_neus_low2_corr, annual_neus_low3_corr, annual_neus_low4_corr, annual_neus_low5_corr]
neusmedlist = [annual_neus_med1_corr, annual_neus_med2_corr, annual_neus_med3_corr, annual_neus_med4_corr, annual_neus_med5_corr]

gomlowlist = [annual_gom_low1_corr, annual_gom_low2_corr, annual_gom_low3_corr, annual_gom_low4_corr, annual_gom_low5_corr]
gommedlist = [annual_gom_med1_corr, annual_gom_med2_corr, annual_gom_med3_corr, annual_gom_med4_corr, annual_gom_med5_corr]

nullhypothesisvalueneus = annual_neus_obs_corr
corrlowneus = np.array(neuslowlist)
corrmedneus = np.array(neusmedlist)
t_obslowneus, p_valuelowneus = ttest_1samp(corrlowneus, nullhypothesisvalueneus)
t_obsmedneus, p_valuemedneus = ttest_1samp(corrmedneus, nullhypothesisvalueneus)

nullhypothesisvaluegom = annual_gom_obs_corr
corrlowgom = np.array(gomlowlist)
corrmedgom = np.array(gommedlist)
t_obslowgom, p_valuelowgom = ttest_1samp(corrlowgom, nullhypothesisvaluegom)
t_obsmedgom, p_valuemedgom = ttest_1samp(corrmedgom, nullhypothesisvaluegom)

t_statistic = scipy.stats.t.ppf(1-0.05, 4)

neus_low_corr_avg = sum(neuslowlist)/len(neuslowlist) 
neus_med_corr_avg = sum(neusmedlist)/len(neusmedlist)

gom_low_corr_avg = sum(gomlowlist)/len(gomlowlist) 
gom_med_corr_avg = sum(gommedlist)/len(gommedlist) 

neus_low_corr_stderror = abs(t_statistic * (statistics.stdev(neuslowlist)/(len(neuslowlist)**.5)))
neus_med_corr_stderror = abs(t_statistic * (statistics.stdev(neusmedlist)/(len(neusmedlist)**.5)))

gom_low_corr_stderror = abs(t_statistic * (statistics.stdev(gomlowlist)/(len(gomlowlist)**.5)))
gom_med_corr_stderror = abs(t_statistic * (statistics.stdev(gommedlist)/(len(gommedlist)**.5)))

##########################################################################
#CREATE CC PLOTS 
##########################################################################
def createccplots1(): 
    fig1, ax1 = plt.subplot() 
    ax1.title('Correlation Coefficient of mean GOM SST and GSI', fontsize='16', pad='20') 
    ax1.errorbar(2, gom_med_corr_avg, yerr=np.array([gom_med_corr_stderror]), fmt='o', capsize=5, c='b')
    ax1.errorbar(1, gom_low_corr_avg, yerr=np.array([gom_low_corr_stderror]), fmt='o', capsize=5, c='r')
    ax1.set_xlim(0,3) 
    ax1.set_ylim(-1,1) 
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Low Corr Avg', 'Med Corr Avg'])
    ax1.set_ylabel('Correlation Coefficient') 
    ax1.axhline(y = annual_gom_obs_corr, color = "black")
    ax1.annotate('Observed correlation', (.05,.5))
    ax1.scatter([1,1,1,1,1], gomlowlist, c='lightcoral', label='Low Resolution',marker='x')
    ax1.scatter([2,2,2,2,2], gommedlist, c='lightblue', label='Med Resolution', marker='x')
    ax1.annotate(f'p-value: {int(str(p_valuelowgom)[:3])}', (.7, -.85), bbox=dict(boxstyle="square,pad=0.3", fc="lightcoral", ec="brown", lw=2))
    ax1.annotate(f'p-value: {int(str(p_valuemedgom)[:3])}', (1.7, -.5), bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="steelblue", lw=2))
    plt.savefig('GOM_Correlation_Coefficients.png', dpi=300, bbox_inches='tight')

def createccplots2(): 
    fig2, ax2 = plt.subplot() 
    ax2.title('Correlation Coefficient of mean NEUS SST and GSI', fontsize='16', pad='20') 
    ax2.errorbar(2, neus_med_corr_avg, yerr=np.array([medcorrerror]), fmt='o', capsize=5, c='b')
    ax2.errorbar(1, neus_low_corr_avg, yerr=np.array([lowcorrerror]), fmt='o', capsize=5, c='r')
    ax2.set_xlim(0,3) 
    ax2.set_ylim(-1,1) 
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Low Corr Avg', 'Med Corr Avg'])
    ax2.set_ylabel('Correlation Coefficient') 
    ax2.axhline(y = annual_neus_obs_corr, color = "black")
    ax2.annotate('Observed correlation', (.05,.5)) 
    ax2.scatter([1,1,1,1,1], neuslowlist, c='lightcoral', label='Low Resolution',marker='x')
    ax2.scatter([2,2,2,2,2], neusmedlist, c='lightblue', label='Med Resolution', marker='x')
    ax2.annotate('p-value: 0.457', (.7, -.4), bbox=dict(boxstyle="square,pad=0.3", fc="lightcoral", ec="brown", lw=2))
    ax2.annotate('p-value: 0.257', (1.7, -.15), bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="steelblue", lw=2))
    plt.savefig('NEUS_Correlation_Coefficients.png', dpi=300, bbox_inches='tight')

##########################################################################
#CREATE GSIPLOTS 
##########################################################################

def creategsiplots3(): 
    fig3, plt.figure(figsize=(10, 6), tight_layout=True)
    gs = gridspec.GridSpec(2, 2, hspace=.25)
    PC = ccrs.PlateCarree()
    _LAND_50M = feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#999999')

    ax3 = fig.add_subplot(gs[0, 0], projection=PC)
    ax3.add_feature(_LAND_50M)
    ax3.plot(obs_ssh_points.lat-360, obs_ssh_points.lon) 
    ax3.add_ticks(ax, xlabelinterval=5)
    ax3.set_extent([-82, -50, 25, 44])
    ax3.set_title('LOW vs. MED Atmospheric Forcing -- AVG')
    ax3.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)

    ax4 = fig.add_subplot(gs[0, 1], projection=PC)
    ax4.add_feature(_LAND_50M)
    ax4.plot(modelgsipointsmed1.lon-360, modelgsipointsmed1, c='r', linewidth='.5', label = 'Medium resolution')
    ax4.plot(modelgsipointslow1.lon-360, modelgsipointslow1, c='b', linewidth='.5', label = 'Low resloution') 
    ax4.plot(modelgsipointsmed2.lon-360, modelgsipointsmed2, c='r', linewidth='.5')
    ax4.plot(modelgsipointslow2.lon-360, modelgsipointslow2, c='b', linewidth='.5') 
    ax4.plot(modelgsipointsmed3.lon-360, modelgsipointsmed3, c='r', linewidth='.5')
    ax4.plot(modelgsipointslow3.lon-360, modelgsipointslow3, c='b', linewidth='.5') 
    ax4.plot(modelgsipointsmed4.lon-360, modelgsipointsmed4, c='r', linewidth='.5')
    ax4.plot(modelgsipointslow4.lon-360, modelgsipointslow4, c='b', linewidth='.5') 
    ax4.plot(modelgsipointsmed5.lon-360, modelgsipointsmed5, c='r', linewidth='.5')
    ax4.plot(modelgsipointslow5.lon-360, modelgsipointslow5, c='b', linewidth='.5')
    ax4.add_ticks(ax, xlabelinterval=5)
    ax4.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)
    ax4.set_extent([-82, -50, 25, 44])
    ax4.set_title('LOW vs MED Atmospheric Forcing -- All ENS')

    plt.savefig('Gulfstream_Visualization.png', dpi=300, bbox_inches='tight')

##########################################################################
#RUN WHEN EXECUTED AS A SCRIPT 
##########################################################################

if name=="__main__": 
    print("CREATING CC PLOT 1") 
    createccplots1()
    print("CREATING CC PLOT 2") 
    createccplots2()
    print("CREATING GSI PLOT") 
    creategsiplots() 
    print("COMPLETE") 
    