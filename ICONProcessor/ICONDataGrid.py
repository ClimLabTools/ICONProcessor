import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc

from datetime import datetime, timedelta
from colorama import Fore
import pytz


class ICONDataGrid:

    @staticmethod
    def convert_float_dt(float_datetime):
        date_part = int(float_datetime)
        time_part = (float_datetime - date_part) * 24 * 3600  # Convert fractional part to seconds
        time_delta = timedelta(seconds=round(time_part))
        datetime_obj = datetime.strptime(str(date_part), '%Y%m%d') + time_delta
        return pytz.utc.localize(datetime_obj)

    def __init__(self, data_file, grid, calc_wind=False):
        self.ds = xr.open_dataset(data_file, autoclose=True, engine='netcdf4')
        self.grid = grid
        # if grid.cells != self.ds['ncells_2'].size:
        #     print(Fore.RED + 'ERROR: Size of grid does not match size of data file!' + Fore.RESET)

        # if requested, calculate wind components
        if calc_wind:
            self.calculate_wind_components()
            self.__calc_wind = True
        else:
            self.__calc_wind = False

    def calculate_wind_components(self):
        # calculate 3D values
        u = self.ds['u']
        v = self.ds['v']

        ws = mpcalc.wind_speed(u, v)
        wd_from = mpcalc.wind_direction(u, v, convention='from')
        wd_to = mpcalc.wind_direction(u, v, convention='to')

        self.__add_to_ds('wind_speed', u.dims, ws.values, u.units, 'Wind speed')
        self.__add_to_ds('wind_dir_from', u.dims, wd_from.values, 'degree', 'Wind from direction')
        self.__add_to_ds('wind_dir_to', u.dims, wd_to.values, 'degree', 'Wind to direction')
        print('Wind Speed and Wind Direction calculated from u and v')

        # calculate surface values
        u = self.ds['u_10m']
        v = self.ds['v_10m']

        ws = mpcalc.wind_speed(u, v)
        wd_from = mpcalc.wind_direction(u, v, convention='from')
        wd_to = mpcalc.wind_direction(u, v, convention='to')

        self.__add_to_ds('wind_speed_10m', u.dims, ws.values, u.units, 'Wind speed in 10m')
        self.__add_to_ds('wind_dir_from_10m', u.dims, wd_from.values, 'degree', 'Wind from direction in 10m')
        self.__add_to_ds('wind_dir_to_10m', u.dims, wd_to.values, 'degree', 'Wind to direction in 10m')
        print('Wind Speed and Wind Direction in 10m calculated from u_10m and v_10m')

    def __add_to_ds(self, var, dims, values, unit, long_name):
        self.ds[var] = (dims, values)
        self.ds[var].attrs['units'] = unit
        self.ds[var].attrs['long_name'] = long_name

    def get_variable_info(self, var):
        try:
            v = self.ds[var]
        except:
            print(Fore.RED + f"ERROR: Variable '{var}' does not exist in dataset!" + Fore.RESET)
            raise Exception('var not found')

        df_dim = pd.DataFrame({'dims': v.dims, 'shape': v.shape})
        info = {'name': v.name,
                'long_name': v.long_name,
                'unit': v.units}
        return df_dim, info

    def get_datetime(self, time):
        return self.convert_float_dt(self.ds.time.values[time])

    def __plot_variable(self, var, time, values, no_zeros=False, height=None, **kwargs):
        dims, info = self.get_variable_info(var)

        gdf = self.grid.get_triangles()
        gdf[var] = values

        if no_zeros:
            gdf[var] = gdf[var].replace(0, np.nan)

        dt = self.get_datetime(time)
        if height is None:
            title = f"'{info['long_name']}' in {info['unit']}\n{dt}"
        else:
            height_dim = dims.iloc[1]
            height_val = self.ds[height_dim['dims']].values[height] # get height values from dim of variable
            title = f"'{info['long_name']}' in {info['unit']}\nheight: {int(height_val)}/{height_dim['shape']}\n{dt}"
        self.grid.plot_single(gdf, var, title, **kwargs)

    def plot_variable_2D(self, var, time, no_zeros=False, **kwargs):
        if len(self.ds[var].dims) == 3:
            # for variables with height dim = 1
            values = self.ds[var][time, 0, :]
        else:
            values = self.ds[var][time, :]
        self.__plot_variable(var, time, values, no_zeros, **kwargs)

    def plot_variable_3D(self, var, time, height, no_zeros=False, **kwargs):
        values = self.ds[var][time, height, :]
        self.__plot_variable(var, time, values, no_zeros, height, **kwargs)

    def get_data_for_contour(self, cell_id, var):
        gridcell_idx = cell_id - 1

        # find correct dimension for z
        z_dim = self.get_z_dim(var)
        if z_dim is None:
            print(Fore.RED + 'ERROR: Cannot determine z dimension!' + Fore.RESET)

        # get time (float & string) and var values
        time = self.ds.time.values
        dt = [self.convert_float_dt(t) for t in time]
        data = self.ds[var][:, :, gridcell_idx].data.T

        # get z data & info (vertical axis)
        if z_dim == 'depth':
            z = self.ds[z_dim][:].data  # depth not cell dependent
        else:
            z = self.ds[z_dim][:, gridcell_idx].data

        return time, dt, z, data, z_dim

    def plot_simple_contour_cell_variable(self, cell_id, var, figsize=(10, 7), cmap='jet', xticks_skip=1):
        try:
            time, dt, z, data, z_dim = self.get_data_for_contour(cell_id, var)
        except:
            print(print(Fore.RED + 'ERROR: Cannot load data for contour plot!' + Fore.RESET))
            return

        # convert time (float to string)
        dt_str = [x.strftime('%d.%m. - %H:%M') for x in dt]

        # get variable/z info
        z_info = self.get_variable_info(z_dim)[1]
        dims, info = self.get_variable_info(var)

        # Create contour plot
        plt.figure(figsize=figsize)

        x_vals = np.arange(data.shape[1])
        contour = plt.pcolormesh(x_vals, z, data, cmap=cmap)
        cbar = plt.colorbar(contour, label=f"{info['long_name']} in {info['unit']}")  # Add colorbar
        plt.xlabel('Time')
        plt.xticks(x_vals[::xticks_skip], dt_str[::xticks_skip], rotation=90)
        plt.ylabel(f"{z_info['long_name']} in {z_info['unit']} ({z_dim})")
        plt.title('Vertical Profile over Time')
        plt.tight_layout()
        plt.show()

    def plot_wind_map(self, time, surface=True, height=-1, barb_distance=5, coords_precision=2,
                      figsize=(10, 10), outlines_gdf=None,
                      cb_range=range(0, 11, 1), barb_increments=dict(half=5, full=10, flag=50), xlim=None, ylim=None,
                      show=True, save_path=None):

        # select right variable
        if surface:
            u = self.ds['u_10m'][time, 0, :]
            v = self.ds['v_10m'][time, 0, :]
        else:
            u = self.ds['u'][time, height, :]
            v = self.ds['v'][time, height, :]

        # calculate wind speed
        ws = mpcalc.wind_speed(u, v)

        # get coords of center points (high precision for high resolutions)
        # lat = np.rad2deg(u.clat.values).round(coords_precision)
        # lon = np.rad2deg(u.clon.values).round(coords_precision)
        lat = np.rad2deg(np.array(self.grid.ds_grid['clat'])).round(coords_precision)
        lon = np.rad2deg(np.array(self.grid.ds_grid['clon'])).round(coords_precision)

        # collect all data in dataframe
        df = pd.DataFrame({'lat': lat, 'lon': lon, 'ws': ws, 'u': u.values, 'v': v.values})

        # for points with same coords -> calc mean (no duplicates if higher precision is chosen but makes DF very large)
        df_grouped = df.groupby(['lat', 'lon'], as_index=False).mean()

        # Get unique latitudes and longitudes
        unique_lats = sorted(df_grouped['lat'].unique())
        unique_lons = sorted(df_grouped['lon'].unique())

        # Create a 2D grid initialized with NaNs
        grid_ws = pd.DataFrame(np.nan, index=unique_lats, columns=unique_lons)
        grid_u = pd.DataFrame(np.nan, index=unique_lats, columns=unique_lons)
        grid_v = pd.DataFrame(np.nan, index=unique_lats, columns=unique_lons)

        # Fill the grid with values from the DataFrame
        for _, row in df_grouped.iterrows():
            grid_ws.at[row['lat'], row['lon']] = row['ws']
            grid_u.at[row['lat'], row['lon']] = row['u']
            grid_v.at[row['lat'], row['lon']] = row['v']

        # convert dataframe to array
        gws = grid_ws.to_numpy()
        gu = grid_u.to_numpy()
        gv = grid_v.to_numpy()

        # distance between barbs
        skip = barb_distance

        # start figure and set axis
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # plot outlines
        if outlines_gdf is not None:
            outlines_gdf.boundary.plot(ax=ax, linewidth=1, edgecolor='black', alpha=0.5)

        # plot wind speed
        cf = ax.contourf(unique_lons, unique_lats, gws, cb_range, cmap=plt.cm.BuPu)
        plt.colorbar(cf, pad=0.01, aspect=30, shrink=0.5, label="Wind Speed in m/s")
        ax.barbs(unique_lons[::skip], unique_lats[::skip], gu[::skip, ::skip], gv[::skip, ::skip],
                 color='black', length=5, alpha=0.5, sizes=dict(emptybarb=0),
                 barb_increments=barb_increments)

        if xlim is None or ylim is None:
            ax.set(xlim=(lon.min(), lon.max()), ylim=(lat.min(), lat.max()))
        else:
            ax.set(xlim=xlim, ylim=ylim)

        if surface:
            height_str = '10m above surface'
        else:
            height_str = f"{int(self.ds['height'].values[height])}/{self.ds['height'].size}"
            # height_str = f"{int(self.ds['plev'].values[height])}/{self.ds['plev'].size}"
        ax.set_title(f'Wind at {self.get_datetime(time)} UTC\nHeight: {height_str}\nBarb increments (m/s): '
                     f'{barb_increments["half"]} (half), '
                     f'{barb_increments["full"]} (full), '
                     f'{barb_increments["flag"]} (flag)')

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def get_model_layers(self, cell_id):
        z_ifc = self.ds['z_ifc'][:, cell_id - 1].data
        df_layers = pd.DataFrame()
        for i in range(1, len(z_ifc)):
            line = {'id': i,
                    'high': z_ifc[i - 1],
                    'low': z_ifc[i]}
            df_layers = pd.concat([df_layers, pd.DataFrame(line, index=[i])])
        return df_layers

    def get_z_dim(self, var):
        z_var_dim = self.get_variable_info(var)[0].iloc[1]

        # check full level center (z_mc)
        z_mc_dim = self.get_variable_info('z_mc')[0].iloc[0]
        if z_var_dim['dims'] == z_mc_dim['dims']:
            return 'z_mc'

        # check half level center (i_ifc)
        z_ifc_dim = self.get_variable_info('z_ifc')[0].iloc[0]
        if z_var_dim['dims'] == z_ifc_dim['dims']:
            return 'z_ifc'

        # check depth
        z_depth_dim = self.get_variable_info('depth')[0].iloc[0]
        if z_var_dim['dims'] == z_depth_dim['dims']:
            return 'depth'

        # nothing found yet comparing names -> try with shape

        # check full level center (z_mc)
        if z_var_dim['shape'] == z_mc_dim['shape']:
            return 'z_mc'

        # check half level center (i_ifc)
        z_ifc_dim = self.get_variable_info('z_ifc')[0].iloc[0]
        if z_var_dim['shape'] == z_ifc_dim['shape']:
            return 'z_ifc'

        return None