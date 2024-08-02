import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import xarray
import numpy as np
import pandas as pd
import os
import sys
import datetime
from pathlib import Path 

sys.path.append('/home/Harris.Longfield/CEFI-regional-MOM6/diagnostics/physics/')
from plot_common import add_ticks, autoextend_colorbar, corners, annotate_skill, open_var, save_figure


def makechlplot(start, end, label1, label2): 

    ###########################################################################################################################
    #FILE PATHS 
    ###########################################################################################################################

    erasrunpath = Path('/home/Andrew.C.Ross/git/nwa12/experiments/bgc_forecasts/data/analysis_chlos_climatology.nc')
    erasrun = xarray.open_mfdataset(erasrunpath)

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

    ###########################################################################################################################
    #DEFINE MASKS
    ###########################################################################################################################

    masks = xarray.open_dataset('/home/Andrew.C.Ross/git/nwa12/data/geography/masks/regions_30m.nc')

    weightsgom = masks['areacello'].where(masks['GOM']).fillna(0)
    weightsmab = masks['areacello'].where(masks['MAB']).fillna(0)

    ###########################################################################################################################
    #OPEN VARS 
    ###########################################################################################################################

    chlos_med1 = open_var(rootmed1, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 # kg m-3 -> mg m-3
    chlos_med2 = open_var(rootmed2, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 # kg m-3 -> mg m-3
    chlos_med3 = open_var(rootmed3, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 # kg m-3 -> mg m-3
    chlos_med4 = open_var(rootmed4, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 # kg m-3 -> mg m-3
    chlos_med5 = open_var(rootmed5, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 # kg m-3 -> mg m-3

    chlos_low1 = open_var(rootlow1, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 
    chlos_low2 = open_var(rootlow2, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 
    chlos_low3 = open_var(rootlow3, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 
    chlos_low4 = open_var(rootlow4, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 
    chlos_low5 = open_var(rootlow5, 'ocean_cobalt_daily_2d', 'chlos') * 1e6 

    ###########################################################################################################################
    #MASK OVER REGIONS
    ###########################################################################################################################

    mean_chlos_med1 = chlos_med1.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_med2 = chlos_med2.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_med3 = chlos_med3.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_med4 = chlos_med4.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_med5 = chlos_med5.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()

    mean_chlos_low1 = chlos_low1.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_low2 = chlos_low2.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_low3 = chlos_low3.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_low4 = chlos_low4.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()
    mean_chlos_low5 = chlos_low5.sel(slice(1997,2017)).weighted(weights).mean(['yh', 'xh']).compute()

    mean_eras_chlos = erasrun.weighted(weights).mean(['yh', 'xh']).compute() 

    mean_chlos_med1mab = chlos_med1.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_med2mab = chlos_med2.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_med3mab = chlos_med3.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_med4mab = chlos_med4.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_med5mab = chlos_med5.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()

    mean_chlos_low1mab = chlos_low1.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_low2mab = chlos_low2.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_low3mab = chlos_low3.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_low4mab = chlos_low4.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()
    mean_chlos_low5mab = chlos_low5.sel(slice(1997,2017)).weighted(weightsmab).mean(['yh', 'xh']).compute()

    mean_eras_chlosmab = erasrun.weighted(weightsmab).mean(['yh', 'xh']).compute()     
    
    ###########################################################################################################################
    #MAKE CLIMATOLOGIES 
    ###########################################################################################################################

    chlos_med1_climatology = mean_chlos_med1.groupby('time.dayofyear').mean()
    chlos_med2_climatology = mean_chlos_med2.groupby('time.dayofyear').mean()
    chlos_med3_climatology = mean_chlos_med3.groupby('time.dayofyear').mean()
    chlos_med4_climatology = mean_chlos_med4.groupby('time.dayofyear').mean()
    chlos_med5_climatology = mean_chlos_med5.groupby('time.dayofyear').mean()

    chlos_low1_climatology = mean_chlos_low1.groupby('time.dayofyear').mean() 
    chlos_low2_climatology = mean_chlos_low2.groupby('time.dayofyear').mean() 
    chlos_low3_climatology = mean_chlos_low3.groupby('time.dayofyear').mean() 
    chlos_low4_climatology = mean_chlos_low4.groupby('time.dayofyear').mean() 
    chlos_low5_climatology = mean_chlos_low5.groupby('time.dayofyear').mean() 

    eras_climatology = mean_eras_chlos.groupby('dayofyear', squeeze=False).mean() 

    chlos_med1_climatology_mab = mean_chlos_med1mab.groupby('time.dayofyear').mean()
    chlos_med2_climatology_mab = mean_chlos_med2mab.groupby('time.dayofyear').mean()
    chlos_med3_climatology_mab = mean_chlos_med3mab.groupby('time.dayofyear').mean()
    chlos_med4_climatology_mab = mean_chlos_med4mab.groupby('time.dayofyear').mean()
    chlos_med5_climatology_mab = mean_chlos_med5mab.groupby('time.dayofyear').mean()

    chlos_low1_climatology_mab = mean_chlos_low1mab.groupby('time.dayofyear').mean() 
    chlos_low2_climatology_mab = mean_chlos_low2mab.groupby('time.dayofyear').mean() 
    chlos_low3_climatology_mab = mean_chlos_low3mab.groupby('time.dayofyear').mean() 
    chlos_low4_climatology_mab = mean_chlos_low4mab.groupby('time.dayofyear').mean() 
    chlos_low5_climatology_mab = mean_chlos_low5mab.groupby('time.dayofyear').mean() 

    eras_climatology_mab = mean_eras_chlosmab.groupby('dayofyear', squeeze=False).mean() 

    ###########################################################################################################################
    #MAKE PLOTS
    ###########################################################################################################################
    
    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    gs = gridspec.GridSpec(2, 2, hspace=.25)

        ###########################################################################################################################
        #PLOT1 
        ###########################################################################################################################

    x = pd.date_range('2004-01-01', freq='1D', periods=len(chlos_med1_climatology))

    ax1 = fig.add_subplot(gs[:, 0])
    ax1 = plt.subplots(figsize=(12 * 1.2, 6))  # 120% wider than 12 inches wide
    
    ax1.plot(x, chlos_med1_climatology, c='lightblue', label = 'Medium')
    ax1.plot(x, chlos_med2_climatology, c='lightblue')
    ax1.plot(x, chlos_med3_climatology, c='lightblue')
    ax1.plot(x, chlos_med4_climatology, c='lightblue')
    ax1.plot(x, chlos_med5_climatology, c='lightblue')

    ax1.plot(x, chlos_low1_climatology, c='lightcoral', label = 'Low')
    ax1.plot(x, chlos_low2_climatology, c='lightcoral')
    ax1.plot(x, chlos_low3_climatology, c='lightcoral')
    ax1.plot(x, chlos_low4_climatology, c='lightcoral')
    ax1.plot(x, chlos_low5_climatology, c='lightcoral')

    ax1.plot(x, eras_climatology, c='black', label='ERA5') 

    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none') 
    ax1.spines['bottom'].set_linewidth(2) 
    ax1.spines['left'].set_linewidth(2) 

    ax1.tick_params(axis='x', bottom=False, top=False) 
    ax1.tick_params(axis='y', left=False, right=False) 

    ax1.legend(loc = 'lower center', frameon=False, ncol=2)

    ax1.set_title('GOM') 
    
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0,1.8)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    plt.rcParams['figure.dpi'] = 1200
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ###########################################################################################################################
        #PLOT2
        ###########################################################################################################################

    ax2 = fig.add_subplot(gs[:, 1])

    ax2.plot(x, chlos_med1_climatology_mab, c='lightblue', label = 'Medium')
    ax2.plot(x, chlos_med2_climatology_mab, c='lightblue')
    ax2.plot(x, chlos_med3_climatology_mab, c='lightblue')
    ax2.plot(x, chlos_med4_climatology_mab, c='lightblue')
    ax2.plot(x, chlos_med5_climatology_mab, c='lightblue')

    ax2.plot(x, chlos_low1_climatology_mab, c='lightcoral', label = 'Low')
    ax2.plot(x, chlos_low2_climatology_mab, c='lightcoral')
    ax2.plot(x, chlos_low3_climatology_mab, c='lightcoral')
    ax2.plot(x, chlos_low4_climatology_mab, c='lightcoral')
    ax2.plot(x, chlos_low5_climatology_mab, c='lightcoral')

    ax2.plot(x, eras_climatology_mab, c='black', label='ERA5') 

    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none') 
    ax2.spines['bottom'].set_linewidth(2) 
    ax2.spines['left'].set_linewidth(2) 

    ax2.tick_params(axis='x', bottom=False, top=False) 
    ax2.tick_params(axis='y', left=False, right=False) 

    ax2.legend(loc = 'lower center', frameon=False, ncol=2)

    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0,1.8)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    ax2.set_title('MAB') 

    plt.rcParams['figure.dpi'] = 1200
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    save_figure('chlos_compare', label=label, pdf=True)


###########################################################################################################################
#RUN AS SCRIPT 
###########################################################################################################################

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('start_year', default = '1997', nargs = '?', type = str)
    parser.add_argument('end_year', default = '2009', nargs = '?', type = str)
    parser.add_argument('-l', '--label', help='Label to add to figure file names', type=str, default='')
    parser.add_argument('-l2', '--label2', help='Label to add to figure file names', type=str, default='')
    args = parser.parse_args() 
    makechlplot(args.start_year, args.end_year, args.label, args.label2)

