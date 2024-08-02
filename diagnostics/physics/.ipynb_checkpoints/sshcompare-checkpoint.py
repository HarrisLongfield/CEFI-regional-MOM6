"""
Compare model SSH with 1993--2019 data from GLORYS.
Includes a plot of the Gulf Stream position and index,
and a comparison of time mean SSH.
Uses whatever model data can be found within the directory pp_root,
and does not try to match the model and observed time periods.
How to use:
python sshcompare.py /archive/acr/fre/NWA/2023_04/NWA12_COBALT_2023_04_kpo4-coastatten-physics/gfdl.ncrc5-intel22-prod /work/hnl/spear_lo_hist_ens_01
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

def plot_ssh_eval(pp_root, label, pp_root2, label2):
    
    #SOLVING FOR MODEL 1 

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

    #SOLVING FOR MODEL 2 

    model2_ssh = xarray.open_mfdataset('/work/hnl/spear_lo_hist_ens_01/ocean.200101-201012.ssh.nc')['ssh']
    model2_ssh['time'] = model2_ssh.indexes['time'].to_datetimeindex()
    
    model2ssh_ave = model2_ssh.mean('time')

    model2_grid = xarray.open_dataset('/work/hnl/spear_lo_hist_ens_01/ocean_z.static.nc')

    model2_ssh_index, model2_ssh_points = compute_gs(model2_ssh, data_grid=model2_grid[['geolon', 'geolat']].rename({'geolon': 'lon', 'geolat': 'lat'}))
    
    model2_rolled = model2_ssh_index.rolling(25, center=True, min_periods=25).mean().dropna()

    #PLOTTING MODEL 1 

    ax = fig.add_subplot(gs[0, :])
    (model_rolled - satellite_rolled).plot(ax=ax, c='k', label = f'MOM6 Model rolling mean - Satellite rolling mean')
    (model_ssh_index - satellite_ssh_index).plot(ax=ax, color='blue', label = f'MOM6 Model index- Satellite indexx')
    ax.set_title('MOM6 Model - Satellite')
    ax.set_xlabel('')
    ax.set_ylabel('Difference in rolling Index (positive: model further north)')
    ax.set_ylim(-3,3)
    ax.axhline(y = 0, color = "red")
    ax.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)

    save_figure('gulfstream_compare2.2', label=label, pdf=True)

    #PLOTTING MODEL 2 

    ax = fig.add_subplot(gs[1, :])
    (model2_rolled - satellite_rolled).plot(ax=ax, c='k', label = f'SPEAR Model rolling mean - Satellite rolling mean')
    (model2_ssh_index - satellite_ssh_index).plot(ax=ax, color='blue', label = f'SPEAR Model index- Satellite indexx')
    ax.set_title('SPEAR Model - Satellite')
    ax.set_xlabel('')
    ax.set_ylabel('Difference in rolling Index (positive: model further north)')
    ax.set_ylim(-3,3)
    ax.axhline(y = 0, color = "red")
    ax.legend(ncol=4, loc='lower right', frameon=False, fontsize=8)

    #save_figure('gulfstream_compare2.2', label=label, pdf=True)
    plt.show() 

###############################################################################################################################
#Code to run the file from the command at the top
###############################################################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pp_root', help='Path to postprocessed data (up to but not including /pp/)')
    parser.add_argument('pp_root2', help='Path to postprocessed data (up to but not including /pp/)')
    parser.add_argument('-l', '--label', help='Label to add to figure file names', type=str, default='')
    parser.add_argument('-l2', '--label2', help='Label to add to figure file names', type=str, default='')
    args = parser.parse_args()
    plot_ssh_eval(args.pp_root, args.label, args.pp_root2, args.label2) 



