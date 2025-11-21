import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import metpy.calc as mpcalc

from matplotlib.colors import Normalize, ListedColormap
from metpy.units import units
from suntimes import SunTimes


class ICONMeteogram:

    def __init__(self, path, prefix):
        self.path = path
        self.files = []

        self.sfc_var_p = 16
        self.sfc_var_temp = 24
        self.sfc_var_u = 26
        self.sfc_var_v = 27
        self.sfc_var_gust = 29
        self.var_cc = 19        # cloud cover

        for file in sorted(os.listdir(path)):
            if file.startswith(prefix):
                self.files.append(path + '/' + file)

        if len(self.files) > 0:
            print('Meteogram files found: ', len(self.files))

            # get first and last dataset for meta info
            self.ds_first = xr.load_dataset(self.files[0])
            self.ds_last = xr.load_dataset(self.files[-1])
            self.dt_first = pd.to_datetime(self.ds_first['date'].values[0].decode("utf-8"))
            self.dt_last = pd.to_datetime(self.ds_last['date'].values[-1].decode("utf-8"))
            print('First record:', self.dt_first)
            print('Last record:', self.dt_last)

            # load variable and location info
            self.vars, self.sfc_vars = self.get_all_variables()
            self.locs = self.get_all_locations()
            print('Locations found:', self.locs.shape[0])
            print(self.locs['name'].to_list())
        else:
            print('No meteogram files found.')

    def get_all_locations(self):
        df = pd.DataFrame()
        for i in range(self.ds_first.nstations.size):
            info = pd.DataFrame([self.get_location_info(i)])
            df = pd.concat([df, info], ignore_index=True)
        return df

    def get_all_variables(self):
        df = pd.DataFrame()
        for i in range(self.ds_first.nvars.size):
            info = pd.DataFrame([self.get_variable_info(i)])
            df = pd.concat([df, info], ignore_index=True)

        df_sfc = pd.DataFrame()
        for i in range(self.ds_first.nsfcvars.size):
            info = pd.DataFrame([self.get_sfc_variable_info(i)])
            df_sfc = pd.concat([df_sfc, info], ignore_index=True)
        return df, df_sfc

    def get_location_info(self, loc_idx):
        location = {'idx': loc_idx,
                    'name': self.ds_first['station_name'].values[loc_idx].decode("utf-8"),
                    'lat': round(self.ds_first['station_lat'].values[loc_idx], 4),
                    'lon': round(self.ds_first['station_lon'].values[loc_idx], 4),
                    'alt': round(self.ds_first['station_hsurf'].values[loc_idx])
                    }
        return location

    def get_variable_info(self, var_idx):
        variable = {'idx': var_idx,
                    'name': self.ds_first['var_name'].values[var_idx].decode("utf-8"),
                    'long_name': self.ds_first['var_long_name'].values[var_idx].decode("utf-8"),
                    'unit': self.ds_first['var_unit'].values[var_idx].decode("utf-8"),
                    'nlevs': self.ds_first['var_nlevs'].values[var_idx],
                    'group_id': self.ds_first['var_group_id'].values[var_idx]
                    }
        return variable

    def get_sfc_variable_info(self, var_idx):
        variable = {'idx': var_idx,
                    'name': self.ds_first['sfcvar_name'].values[var_idx].decode("utf-8"),
                    'long_name': self.ds_first['sfcvar_long_name'].values[var_idx].decode("utf-8"),
                    'unit': self.ds_first['sfcvar_unit'].values[var_idx].decode("utf-8"),
                    'group_id': self.ds_first['sfcvar_group_id'].values[var_idx]
                    }
        return variable

    def plot_meteogram(self, location_idx, save_path=None, overwrite_name=None,
                       dt_from=None, dt_to=None, figsize=(10, 8)):

        # get location name
        if overwrite_name is None:
            name = self.locs['name'].values[location_idx]
        else:
            name = overwrite_name

        # get time range
        if dt_from is None:
            dt_from = self.dt_first
        if dt_to is None:
            dt_to = self.dt_last

        # get location coords
        lon = self.locs['lon'].values[location_idx]
        lat = self.locs['lat'].values[location_idx]
        alt = self.locs['alt'].values[location_idx]

        # get meteogram data for location
        data, cc, cc_heights = self.load_values(location_idx)

        suntimes = SunTimes(longitude=lon,
                            latitude=lat,
                            altitude=alt)

        # START OF PLOT
        fig, axs = plt.subplot_mosaic([['temp'], ['wind'], ['cc']],
                                      constrained_layout=True,
                                      figsize=figsize)

        axs['wind'].sharex(axs['cc'])
        axs['temp'].sharex(axs['wind'])

        # plot location/time data above first row
        ax = axs['temp']
        ax.set_title(f'Hintereisferner {name}: {lon}°, {lat}° ({alt}m)', loc='left', size=8)
        ax.set_title(f'from {dt_from.strftime("%d.%m.%y %H:%M")} to {dt_to.strftime("%d.%m.%y %H:%M")} UTC',
                     loc='right', size=8)

        # -> TEMP (1st row; left)
        self.__add_sunhours(ax, data['date'], suntimes)
        ax.set_ylim(data['temp'].min() - 2, data['temp'].max() + 2)
        ax.plot(data['date'], data['temp'], color='#f30000', alpha=1, linestyle='-', linewidth='2', zorder=4)
        ax.fill_between(data['date'], data['temp'], color='#f30000', alpha=0.1)
        ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='gray', linewidth=0.4)
        ax.tick_params(axis="x", direction="in", pad=-12, which='both')
        ax.xaxis.set_tick_params(labelbottom=False, which='both')
        ax.set_ylabel('Temperature (°C)', color='red')

        # -> PRESSURE (1st row; right)
        axr = ax.twinx()
        axr.plot(data['date'], data['p'], color='green', alpha=1, linestyle='dotted', linewidth=2)
        axr.set_ylim(data['p'].max() + 2, data['p'].min() - 2)
        axr.invert_yaxis()
        axr.set_ylabel('Pressure (hPa)', color='green')

        # -> WIND (2nd row)
        ax = axs['wind']
        self.__add_sunhours(ax, data['date'], suntimes)
        ax.set_ylim(0, data['gust'].max() + 6)
        ax.plot(data['date'], data['wspd'], color='#3f91ff', alpha=1, linewidth=2)
        ax.fill_between(data['date'], data['wspd'], color='#3f91ff', alpha=0.1)
        ax.plot(data['date'], data['gust'], color='#379696', linestyle='-', alpha=1, linewidth=1.5)
        ax.barbs(data['date'].values[::6], data['gust'].max() + 3, data['u'].values[::6], data['v'].values[::6],
                 length=5, alpha=0.7, pivot='middle',
                 barb_increments=dict(half=1, full=2, flag=6), zorder=9)
        ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='gray', linewidth=0.4)
        ax.tick_params(axis="x", direction="in", pad=-12, which='both')
        ax.xaxis.set_tick_params(labelbottom=False, which='both')
        ax.set_ylabel('Wind speed (m/s)', color='#3f91ff')
        axr = ax.twinx() # wind gust label on 2nd y axis
        axr.set_ylabel('\nWind gust (m/s)', color='#379696')
        axr.tick_params(axis='y', which='both', left=False, right=False, labelright=False)

        # -> CLOUD COVER (3rd row)
        colors = [(1, 1, 1, 0), (0.4, 0.4, 0.4, 1)]  # RGBA for transparent and dark grey
        cmap = ListedColormap(colors)
        xs = mpl.dates.date2num(data['date'])

        ax = axs['cc']

        self.__add_sunhours(ax, data['date'], suntimes)
        ax.imshow(cc, extent=[xs[0], xs[-1], len(cc_heights)-1, 0],
                  aspect='auto', cmap=cmap, alpha=1, zorder=9,
                  interpolation='nearest')
        ticks = np.arange(0, len(cc_heights)-1, 10)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(cc_heights[:-1:10], 1))
        ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='gray', linewidth=0.4)
        ax.set_ylabel('Altitude (km)')

        # Format the x-axis
        ax.set_xlim(dt_from, dt_to)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))

        # Customize the ticks
        ax.tick_params(axis='x', which='major', size=8, width=1, color='black')
        ax.tick_params(axis='x', which='minor', direction="in")

        plt.suptitle('METEOGRAM', fontsize=16)

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()


    def load_values(self, location_idx):
        df_data = pd.DataFrame()
        cc = None

        # process all individual meteogram files and concatenate data
        for file in self.files:
            ds = xr.load_dataset(file)

            df_tmp = pd.DataFrame()
            df_tmp['date'] = ds['date'].values

            # get surface variables
            df_tmp['p'] = ds['sfcvalues'].values[:, self.sfc_var_p, location_idx] / 100
            df_tmp['temp'] = ds['sfcvalues'].values[:, self.sfc_var_temp, location_idx] - 273.15
            df_tmp['u'] = ds['sfcvalues'].values[:, self.sfc_var_u, location_idx]
            df_tmp['v'] = ds['sfcvalues'].values[:, self.sfc_var_v, location_idx]
            df_tmp['gust'] = ds['sfcvalues'].values[:, self.sfc_var_gust, location_idx]
            df_data = pd.concat([df_data, df_tmp], ignore_index=True)

            # get 2D variables
            cc_temp = np.array(ds['values'].values[:, :, self.var_cc, location_idx].T)
            if cc is None:
                cc = cc_temp
            else:
                cc = np.concatenate((cc, cc_temp), axis=1)

        df_data['date'] = pd.to_datetime(df_data['date'].str.decode("utf-8"))

        # Calculate wind speed and direction
        df_data['wdir'] = mpcalc.wind_direction(df_data['u'].values * units('m/s'), df_data['v'].values * units('m/s'))
        df_data['wspd'] = mpcalc.wind_speed(df_data['u'].values * units('m/s'), df_data['v'].values * units('m/s'))

        # Get heights for cloud cover (in km)
        cc_heights = self.ds_first['heights'].values[:, self.var_cc, location_idx] / 1000

        return df_data, cc, cc_heights

    def __add_sunhours(self, ax, datetimes, suntimes):
        previous_day = None

        # Iterate through the list of datetimes
        for current_datetime in datetimes:
            current_day = current_datetime.date()
            if current_day != previous_day:
                sunrise = suntimes.risewhere(current_day, 'UTC')
                sunset = suntimes.setwhere(current_day, 'UTC')

                # add to plot
                ax.axvspan(sunrise, sunset, facecolor='#f0f8c5', alpha=.8, zorder=0)
                ax.axvline(current_day, linewidth=0.8, color='lightgrey')
                previous_day = current_day