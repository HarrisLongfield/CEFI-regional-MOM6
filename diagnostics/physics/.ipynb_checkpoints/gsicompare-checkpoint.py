"""
Compare model SSH with 1993--2019 data from GLORYS.
Includes a plot of the Gulf Stream position and index,
and a comparison of time mean SSH.
Uses whatever model data can be found within the directory pp_root,
and does not try to match the model and observed time periods.
How to use:
python gsicompare.py /archive/acr/fre/NWA/2023_04/NWA12_COBALT_2023_04_kpo4-coastatten-physics/gfdl.ncrc5-intel22-prod /work/hnl/spear_lo_hist_ens_01
python gsicompare.py /archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_1/gfdl.ncrc5-intel22-prod /work/hnl/spear_lo_hist_ens_01  
"""

###############################################################################################################################
#Import Packages 
###############################################################################################################################

import cartopy.feature as feature
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import datetime
import numpy as np
import pandas as pd
import xarray
import os 
import xesmf
from plot_common import autoextend_colorbar, corners, get_map_norm, open_var, add_ticks, annotate_skill, save_figure

###############################################################################################################################
#MISC 
###############################################################################################################################

_LAND_50M = feature.NaturalEarthFeature(
    'physical', 'land', '50m',
    edgecolor='face',
    facecolor='#999999'
)

PC = ccrs.PlateCarree()

###############################################################################################################################
#Compute the index of the gulf stream (float) using ssh data and a datagrid 
###############################################################################################################################

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

###############################################################################################################################
#Create the figures for plotting. Figure 1 compares the first model to 
###############################################################################################################################

def plot_ssh_eval(pp_root, label, pp_root2, label2, start_date, end_date):
    
    ############################################################################################################
    #EVALUATE DATES
    ############################################################################################################
    
    if start_date[0] == '0': 
        strtmo = int(start_date[1]) 
    else: 
        strtmo = int(start_date[:2]) 

    if end_date[0] == '0': 
        endmo = int(end_date[1]) 
    else: 
        endmo = int(end_date[:2]) 

    if start_date[3] == '0': 
        strtday = int(start_date[4]) 
    else: 
        strtday = int(start_date[3:5]) 

    if end_date[3] == '0': 
        endday = int(end_date[4]) 
    else: 
        endday = int(end_date[3:5])  

    
    start_dater = datetime.date(int(start_date[6::]), strtmo, strtday) 
    end_dater = datetime.date(int(end_date[6::]), endmo, endday)

    range_mo = 12 * (end_dater.year - start_dater.year) + end_dater.month - start_dater.month
    
    ############################################################################################################
    #SOLVING FOR MODEL 1 
    ############################################################################################################

    # Ideally would use SSH, but some diag_tables only saved zos
    try:
        model_ssh = open_var(pp_root, 'ocean_monthly', 'ssh')
    except:
        print('Using zos')
        model_ssh = open_var(pp_root, 'ocean_monthly', 'zos')
    model_thetao = open_var(pp_root, 'ocean_monthly_z', 'thetao')

    if '01_l' in model_thetao.coords:
        model_thetao = model_thetao.rename({'01_l': 'z_l'})
        
    model_grid = xarray.open_dataset('../data/geography/ocean_static.nc')
    model_ssh_ave = model_ssh.mean('time')
    model_t200 = model_thetao.interp(z_l=200).mean('time')

    model_ssh_index, model_ssh_points = compute_gs(
        model_ssh,
        data_grid=model_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'})
    )

    satellite_ssh_points = xarray.open_dataset('../data/obs/satellite_ssh_points.nc')
    satellite_ssh_index = pd.read_pickle('../data/obs/satellite_ssh_index.pkl')

    model_rolled = model_ssh_index.rolling(25, center=True, min_periods=25).mean().dropna()
    satellite_rolled = satellite_ssh_index.rolling(25, center=True, min_periods=25).mean().dropna()

    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    gs = gridspec.GridSpec(2, 2, hspace=.25)

    ############################################################################################################
    #SOLVING FOR MODEL 2 
    ############################################################################################################

    model2_ssh_datasetnames = ['ocean.200101-201012.ssh.nc', 'ocean.199101-200012.ssh.nc'] #List all ssh data here

    model2_ssh_datasets = ['/work/hnl/spear_lo_hist_ens_01/' + str(name) for name in model2_ssh_datasetnames] # add file path

    if len(model2_ssh_datasets) > 1:
        datasets = xarray.open_mfdataset(model2_ssh_datasets, combine='nested', concat_dim='time')
        model2_ssh = datasets['ssh']
    else:
        model2_ssh = xarray.open_dataset(model2_ssh_datasets[0])['ssh']
    
    model2_ssh['time'] = model2_ssh.indexes['time'].to_datetimeindex()
    
    model2ssh_ave = model2_ssh.mean('time')

    model2_grid = xarray.open_dataset('/work/hnl/spear_lo_hist_ens_01/ocean_z.static.nc')

    model2_ssh_index, model2_ssh_points = compute_gs(model2_ssh, data_grid=model2_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
    
    model2_rolled = model2_ssh_index.rolling(25, center=True, min_periods=25).mean().dropna()

    ############################################################################################################
    #ERROR ANALYSIS 
    ############################################################################################################

    index_error_1 = np.sum(np.sqrt((model_ssh_index - satellite_ssh_index)*(model_ssh_index - satellite_ssh_index)))/range_mo 

    rolling_error_1 = np.sum(np.sqrt((model_rolled - satellite_rolled)*(model_rolled - satellite_rolled)))/range_mo

    index_error_2 = np.sum(np.sqrt((model2_ssh_index - satellite_ssh_index)*(model2_ssh_index - satellite_ssh_index)))/range_mo

    rolling_error_2 = np.sum(np.sqrt((model2_rolled - satellite_rolled)*(model2_rolled - satellite_rolled)))/range_mo

    ############################################################################################################
    #PLOTTING MODEL 1 
    ############################################################################################################

    ax = fig.add_subplot(gs[0, :])
    (model_rolled - satellite_rolled).plot(ax=ax, c='k', label = f'MOM6 vs. Observed (Rolling)')
    (model_ssh_index - satellite_ssh_index).plot(ax=ax, color='blue', alpha = .3, label = f'MOM6 vs. Observed (Index)')
    ax.set_title('MOM6 vs. Observed (Normalized to 0)')
    ax.text(datetime.date(1995, 1, 1), 2.2 ,'Average Index error = ' + str(round(index_error_1,3)))
    ax.text(datetime.date(1995, 1, 1), 1.6 ,'Average Rolling error = ' + str(round(rolling_error_1,3)))
    ax.set_xlabel('')
    ax.set_ylim(-3,3)
    ax.set_xlim(start_dater, end_dater) 
    ax.axhline(y = 0, color = "red")
    ax.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)

    save_figure('gulfstream_compare2.3', label=label, pdf=True)

    ############################################################################################################
    #PLOTTING MODEL 2 
    ############################################################################################################

    ax = fig.add_subplot(gs[1, :])
    (model2_rolled - satellite_rolled).plot(ax=ax, c='k', label = f'SPEAR vs. Observed (Rolling)')
    (model2_ssh_index - satellite_ssh_index).plot(ax=ax, color='blue', alpha = .3, label = f'SPEAR vs. Observed (Index)')
    ax.set_title('SPEAR vs Observed (Normalized to 0)')
    ax.text(datetime.date(1995, 1, 1), 2.2 ,'Average Index error = ' + str(round(index_error_2,3))) 
    ax.text(datetime.date(1995, 1, 1), 1.6 ,'Average Rolling error = ' + str(round(rolling_error_2,3)))  
    ax.set_xlabel('')
    ax.set_ylim(-3,3)
    ax.set_xlim(start_dater, end_dater) 
    ax.axhline(y = 0, color = "red")
    ax.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)

    save_figure('gulfstream_compare2.3', label=label, pdf=True)

###############################################################################################################################
#Code to run the file from the command at the top
###############################################################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pp_root', help='Path to postprocessed data (up to but not including /pp/)')
    parser.add_argument('pp_root2', help='Path to postprocessed data (up to but not including /pp/)')
    parser.add_argument('start_date', default = '01/01/1994', nargs = '?', type = str)
    parser.add_argument('end_date', default = '10/30/2009', nargs = '?', type = str)
    parser.add_argument('-l', '--label', help='Label to add to figure file names', type=str, default='')
    parser.add_argument('-l2', '--label2', help='Label to add to figure file names', type=str, default='')
    args = parser.parse_args() 
    plot_ssh_eval(args.pp_root, args.label, args.pp_root2, args.label2, args.start_date, args.end_date)
