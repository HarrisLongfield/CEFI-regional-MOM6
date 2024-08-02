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
import xgcm
import os 
from glob import glob
from plot_common import autoextend_colorbar, corners, get_map_norm, open_var, add_ticks, annotate_skill, save_figure
import statistics


static = (
    xarray.open_dataset('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_1/gfdl.ncrc5-intel22-prod/pp/ocean_monthly/ocean_monthly.static.nc')
    .squeeze()
    .drop_vars('time', errors = 'ignore')
)


xgrid = xgcm.Grid(
    static,
    coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
    },
    metrics={
        ('X', ): ['dxt'],
        ('Y', ): ['dyt'], 
        ('X', 'Y'): ['areacello']
    },
    periodic=[],
    boundary='fill'
)

rootmed1 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_1/gfdl.ncrc5-intel22-prod') 
rootmed2 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_2/gfdl.ncrc5-intel22-prod') 
rootmed3 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_3/gfdl.ncrc5-intel22-prod') 
rootmed4 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_4/gfdl.ncrc5-intel22-prod') 
rootmed5 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_medatm_ens_5/gfdl.ncrc5-intel22-prod') 

rootlow1 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_1/gfdl.ncrc5-intel22-prod')
rootlow2 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_2/gfdl.ncrc5-intel22-prod')
rootlow3 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_3/gfdl.ncrc5-intel22-prod')
rootlow4 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_4/gfdl.ncrc5-intel22-prod')
rootlow5 = Path('/archive/Harris.Longfield/fre/NWA/2024_06/NWA12_COBALT_2024_06_lowatm_ens_5/gfdl.ncrc5-intel22-prod')

rooteras = Path('/archive/acr/fre/NWA/2024_04_update/NWA12_COBALT_2024_04_nudgets-90d/gfdl.ncrc5-intel22-prod')


tauuomed1 = open_var(rootmed1, 'ocean_monthly', 'tauuo') 
tauuomed2 = open_var(rootmed2, 'ocean_monthly', 'tauuo') 
tauuomed3 = open_var(rootmed3, 'ocean_monthly', 'tauuo') 
tauuomed4 = open_var(rootmed4, 'ocean_monthly', 'tauuo') 
tauuomed5 = open_var(rootmed5, 'ocean_monthly', 'tauuo') 

tauuolow1 = open_var(rootlow1, 'ocean_monthly', 'tauuo') 
tauuolow2 = open_var(rootlow2, 'ocean_monthly', 'tauuo') 
tauuolow3 = open_var(rootlow3, 'ocean_monthly', 'tauuo') 
tauuolow4 = open_var(rootlow4, 'ocean_monthly', 'tauuo') 
tauuolow5 = open_var(rootlow5, 'ocean_monthly', 'tauuo') 

tauvomed1 = open_var(rootmed1, 'ocean_monthly', 'tauvo') 
tauvomed2 = open_var(rootmed2, 'ocean_monthly', 'tauvo') 
tauvomed3 = open_var(rootmed3, 'ocean_monthly', 'tauvo') 
tauvomed4 = open_var(rootmed4, 'ocean_monthly', 'tauvo') 
tauvomed5 = open_var(rootmed5, 'ocean_monthly', 'tauvo') 

tauvolow1 = open_var(rootlow1, 'ocean_monthly', 'tauvo') 
tauvolow2 = open_var(rootlow2, 'ocean_monthly', 'tauvo') 
tauvolow3 = open_var(rootlow3, 'ocean_monthly', 'tauvo') 
tauvolow4 = open_var(rootlow4, 'ocean_monthly', 'tauvo') 
tauvolow5 = open_var(rootlow5, 'ocean_monthly', 'tauvo') 


masks = xarray.open_dataset('/home/Andrew.C.Ross/git/nwa12/data/geography/masks/regions_30m.nc')

weights = masks['areacello'].where(masks['GOM']).fillna(0)

tauuomed1gom = tauuomed1.weighted(weights).mean(['yh', 'xh']).compute()
tauuomed2gom = tauuomed2.weighted(weights).mean(['yh', 'xh']).compute()
tauuomed3gom = tauuomed3.weighted(weights).mean(['yh', 'xh']).compute()
tauuomed4gom = tauuomed4.weighted(weights).mean(['yh', 'xh']).compute()
tauuomed5gom = tauuomed5.weighted(weights).mean(['yh', 'xh']).compute()

tauuolow1gom = tauuolow1.weighted(weights).mean(['yh', 'xh']).compute()
tauuolow2gom = tauuolow2.weighted(weights).mean(['yh', 'xh']).compute()
tauuolow3gom = tauuolow3.weighted(weights).mean(['yh', 'xh']).compute()
tauuolow4gom = tauuolow4.weighted(weights).mean(['yh', 'xh']).compute()
tauuolow5gom = tauuolow5.weighted(weights).mean(['yh', 'xh']).compute()

tauvomed1gom = tauvomed1.weighted(weights).mean(['yh', 'xh']).compute()
tauvomed2gom = tauvomed2.weighted(weights).mean(['yh', 'xh']).compute()
tauvomed3gom = tauvomed3.weighted(weights).mean(['yh', 'xh']).compute()
tauvomed4gom = tauvomed4.weighted(weights).mean(['yh', 'xh']).compute()
tauvomed5gom = tauvomed5.weighted(weights).mean(['yh', 'xh']).compute()

tauvolow1gom = tauvolow1.weighted(weights).mean(['yh', 'xh']).compute()
tauvolow2gom = tauvolow2.weighted(weights).mean(['yh', 'xh']).compute()
tauvolow3gom = tauvolow3.weighted(weights).mean(['yh', 'xh']).compute()
tauvolow4gom = tauvolow4.weighted(weights).mean(['yh', 'xh']).compute()
tauvolow5gom = tauvolow5.weighted(weights).mean(['yh', 'xh']).compute()

tauuohistgom = open_var(rooteras, 'ocean_monthly', 'tauuo').weighted(weights).mean(['yh', 'xh']).compute()
tauvohistgom = open_var(rooteras, 'ocean_monthly', 'tauvo').weighted(weights).mean(['yh', 'xh']).compute()

meddata = np.concatenate([wsmagmed1.values.flatten(), wsmagmed2.values.flatten(), wsmagmed3.values.flatten(), wsmagmed4.values.flatten(), wsmagmed5.values.flatten()])
lowdata = np.concatenate([wsmaglow1.values.flatten(), wsmaglow2.values.flatten(), wsmaglow3.values.flatten(), wsmaglow4.values.flatten(), wsmaglow5.values.flatten()])


fig, ax = plt.subplots(figsize=(12, 6))

# Plot histograms
ax.hist(lowdata, bins=10000, color='lightcoral', histtype='step')
ax.hist(meddata, bins=10000, color='lightblue', histtype='step')
ax.hist(wsmaghist.values.flatten(), bins=4000, color='black', histtype='step', label='ERA5')

# Customize spines and ticks
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.set_ylim(0, 130000)
ax.set_yticks([40000, 80000, 120000])
ax.set_yticklabels(["40,000", "80,000", "120,000"])

# Legend
ax.legend(loc='upper center', frameon=False, ncol=2)

# Adjust figure DPI
plt.rcParams['figure.dpi'] = 1600

ax.savefig() 
