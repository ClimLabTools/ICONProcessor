import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import metpy.calc as mpcalc
from scipy.interpolate import griddata

from shapely.geometry import Polygon, Point
from datetime import datetime, timedelta
from matplotlib.colors import Normalize, ListedColormap
from metpy.units import units
from colorama import Fore
from suntimes import SunTimes
import pytz

###

class ICONGrid:

    # LU class names based on Global Globcover legend (level 1)
    __lu_class_names = ["irrigated croplands",
                      "rainfed croplands",
                      "mosaic cropland (50-70%) - vegetation (20-50%)",
                      "mosaic vegetation (50-70%) - cropland (20-50%)",
                      "closed broadleaved evergreen forest",
                      "closed broadleaved deciduous forest",
                      "open broadleaved deciduous forest",
                      "closed needleleaved evergreen forest",
                      "open needleleaved deciduous forest",
                      "mixed broadleaved and needleleaved forest",
                      "mosaic shrubland (50-70%) - grassland (20-50%)",
                      "mosaic grassland (50-70%) - shrubland (20-50%)",
                      "closed to open shrubland",
                      "closed to open herbaceous vegetation",
                      "sparse vegetation",
                      "closed to open forest regularly flooded",
                      "closed forest or shrubland permanently flooded",
                      "closed to open grassland regularly flooded",
                      "artificial surfaces",
                      "bare areas",
                      "water bodies",
                      "permanent snow and ice",
                      "undefined"]

    # default colors as defined in GLOBCOVER Products Description and Validation Report
    # https://esdac.jrc.ec.europa.eu/public_path/shared_folder/dataset/GLOBCOVER_Products_Description_Validation_Report_I2.1.pdf
    __lu_colors = [
        "#aaf0f0",  # 0
        "#ffff64",  # 1
        "#dcf064",  # 2
        "#cdcd64",  # 3
        "#016301",  # 4
        "#01a000",  # 5
        "#aac800",  # 6
        "#003c00",  # 7
        "#256400",  # 8
        "#788300",  # 9
        "#8ea000",  # 10
        "#be9600",  # 11
        "#966400",  # 12
        "#ffb431",  # 13
        "#ffebaf",  # 14
        "#00785a",  # 15
        "#019678",  # 16
        "#01dc83",  # 17
        "#c31200",  # 18
        "#fff5d7",  # 19
        "#0046c8",  # 20
        "#ffffff",  # 21
        "#c3c3c3"  # 22
    ]

    @staticmethod
    def save_as_gpkg(gdf, filename, layer):
        gdf.to_file(filename=filename, layer=layer, driver="GPKG")

    @staticmethod
    def get_lu_class_name(lu_class=None):
        if lu_class is None:
            return ICONGrid.__lu_class_names.copy()
        else:
            return ICONGrid.__lu_class_names[lu_class]

    @staticmethod
    def get_lu_colors():
        return ICONGrid.__lu_colors.copy()

    @staticmethod
    def rasterize(gdf, vars, res, interpolate=False):
        # coords of center points
        x, y = gdf['clon'].to_numpy(), gdf['clat'].to_numpy()
        xmin, ymin = np.min(x), np.min(y)
        xmax, ymax = np.max(x), np.max(y)

        if interpolate:
            xc, yc = np.meshgrid(np.arange(xmin + res / 2, xmax, res), np.arange(ymin + res / 2, ymax, res))

            rasters = {}
            for v in vars:
                rasters[v] = griddata(gdf[['clon', 'clat']], gdf[v], (xc, yc), method="linear")
        else:
            # variable values to be rasterized
            vals = {}
            for v in vars:
                vals[v] = gdf[v].to_numpy()

            # rasterize
            xr, yr = (x - xmin) / res, (y - ymin) / res
            xr = xr.astype(int)
            yr = yr.astype(int)

            # create empty rasters
            rasters = {}
            counter = np.zeros((yr.max() + 1, xr.max() + 1))
            for v in vars:
                rasters[v] = counter.copy()

            # sum and count pixel values
            for i in range(len(xr)):
                xw = xr[i]
                yh = yr[i]
                counter[yh, xw] += 1

                for v in vars:
                    rasters[v][yh, xw] += vals[v][i]

            # divide by counter -> mean value
            for v in vars:
                rasters[v] = rasters[v] / counter

        return rasters

    def __init__(self, grid_file, ext_param_file):
        # load .nc files
        self.ds_grid = nc.Dataset(grid_file)
        self.ds_ext = nc.Dataset(ext_param_file)
        self.ext_param_file = ext_param_file

        # construct parent file name and load it
        try:
            grid_file_parts = grid_file.split('.')
            parent_file = f'{grid_file_parts[0]}.parent.{grid_file_parts[1]}'
            self.ds_parent = nc.Dataset(parent_file)
        except:
            print('Parent file not found.')

        # get cells
        self.cells = self.ds_grid.dimensions['cell'].size
        self.gdf_triangles = self.__load_triangles()

    def __load_triangles(self):
        lat = self.ds_grid['clat_vertices'][:, :].data
        lon = self.ds_grid['clon_vertices'][:, :].data

        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

        self.extent = {'lon_min':lon.min(),
                       'lat_min':lat.min(),
                       'lon_max':lon.max(),
                       'lat_max':lat.max(),}

        # Define coordinates for multiple triangles
        triangle_coordinates_list = []
        for i, item in enumerate(lat):
            triangle_coordinates_list.append([(lon[i][0], lat[i][0]), (lon[i][1], lat[i][1]), (lon[i][2], lat[i][2])])

        # Loop through triangle coordinates to create polygons for GeoDataframe
        polygons = []
        for triangle_coords in triangle_coordinates_list:
            # Create a Shapely Polygon from the coordinates
            polygon = Polygon(triangle_coords)
            polygons.append(polygon)

        # Create a GeoDataFrame from the list of Polygon geometries
        geometry = gpd.GeoSeries(polygons)
        gdf = gpd.GeoDataFrame(data=range(1, self.cells + 1), columns=['cell_id'], geometry=geometry, crs="EPSG:4326")
        gdf['clat'] = np.rad2deg(self.ds_grid['clat'][:].data)
        gdf['clon'] = np.rad2deg(self.ds_grid['clon'][:].data)
        return gdf

    def check_consistency(self):
        uuid_grid = self.ds_grid.getncattr('uuidOfHGrid')               # uuid of current grid
        uuid_ext = self.ds_ext.getncattr('uuidOfHGrid')                 # uuid of ext params of grid
        uuid_grid_parent = self.ds_grid.getncattr('uuidOfParHGrid')     # uuid of parent grid that is linked to current grid
        uuid_parent = self.ds_parent.getncattr('uuidOfHGrid')           # uuid of the parent grid

        # compare: grid with ext. params
        if uuid_grid == uuid_ext: print('UUIDs match (grid & ext. param): ', uuid_grid)
        else:
            print(Fore.RED + 'UUIDs do not match!')
            print(f'UUID Grid:\t\t\t{uuid_grid}')
            print(f'UUID ext. param:\t{uuid_ext}' + Fore.RESET)

        # compare: grid with parent grid
        if uuid_grid_parent == uuid_parent: print('UUIDs match (grid parent & parent grid): ', uuid_grid)
        else:
            print(Fore.RED + 'UUIDs do not match!')
            print(f'Parent UUID of Grid:\t\t\t{uuid_grid_parent}')
            print(f'UUID of parent:\t{uuid_parent}' + Fore.RESET)

    def get_triangles(self):
        return self.gdf_triangles

    def get_extent(self):
        return list(self.extent.values())

    def get_topo(self, param='topography_c'):
        return self.ds_ext[param][:].data

    def get_variable(self, param):
        return self.ds_ext[param][:].data

    def get_albedo(self, month, band='vis'):
        if band == 'vis':
            return self.ds_ext['ALB'][month - 1, :]
        elif band == 'nir':
            return self.ds_ext['ALNID'][month - 1, :]
        elif band == 'sw' or band == 'shortwave':
            return self.ds_ext['ALUVD'][month - 1, :]

    def get_lu_fraction(self, lu_class=None):
        return self.ds_ext['LU_CLASS_FRACTION'][lu_class, :].data

    def get_dominant_lu_class(self):
        return np.argmax(self.ds_ext['LU_CLASS_FRACTION'][:, :].data, axis=0)

    def plot_dominant_lu_class(self, title, lu_colors=None, ax=None, outlines=None, outlines_color='black',
                               show=True, save_path=None):

        dom_lu_class = self.get_dominant_lu_class()

        if lu_colors is None:
            lu_colors = self.__lu_colors

        cmap = ListedColormap(lu_colors)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        # Discrete bins centered on each class value
        levels = np.arange(-0.5, len(self.__lu_class_names)+.5, 1)

        # read center lon/lat in radiant
        clon_rad = self.ds_grid['clon'][:].data  # center longitude  / rad
        clat_rad = self.ds_grid['clat'][:].data  # center latitutde  / rad

        # convert to degrees
        clon = np.rad2deg(clon_rad)
        clat = np.rad2deg(clat_rad)

        # plot actual map
        tcf = ax.tricontourf(clon, clat, dom_lu_class, cmap=cmap, levels=levels)

        if outlines is not None:
            outlines.boundary.plot(ax=ax, linewidth=1, edgecolor=outlines_color, alpha=0.5)
            ax.set(xlim=(clon.min(), clon.max()), ylim=(clat.min(), clat.max()))

        # Create a legend
        handles = [plt.Line2D([0], [0],
                              marker='o', color=lu_colors[idx], linestyle='',
                              markersize=8, markeredgecolor='black', markeredgewidth=.3,
                              label=f'{idx} - {val}')
                   for idx, val in enumerate(self.__lu_class_names)]

        legend = ax.legend(handles=handles, title="ICON Land Use Classes",
                           loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

        ax.set_title(title, fontweight = 'bold')
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def get_cells_for_coords(self, location_names, latitudes, longitudes, gdf=None):
        if gdf is None:
            gdf_icon = self.gdf_triangles
        else:
            gdf_icon = gdf

        df_location = pd.DataFrame()
        df_location['name'] = location_names
        df_location['longitude'] = longitudes
        df_location['latitude'] = latitudes

        # create Points for coords and turn it into GeoDataFrame
        geometry = [Point(xy) for xy in zip(df_location['longitude'], df_location['latitude'])]
        gdf_coords = gpd.GeoDataFrame(df_location, geometry=geometry, crs='EPSG:4326')

        # find corresponding grid cells
        return gpd.sjoin(gdf_icon, gdf_coords, how="inner", predicate="contains").drop('index_right', axis=1)

    def plot_single(self, gdf, attribute, title, cmap='jet', show=True, save_path=None, engine='fast',
                    outlines=None, outlines_color='black', contour_levels=20, contour_lines=False, contour_labels=False,
                    ax=None, colorbar=True, v_lim=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

        if engine == 'gpd':
            gdf.plot(attribute, ax=ax, cmap=cmap, legend=True, legend_kwds={"shrink": .7})
            ax.set_title(title)
        else:
            # read center lon/lat in radiant
            clon_rad = self.ds_grid['clon'][:].data  # center longitude  / rad
            clat_rad = self.ds_grid['clat'][:].data  # center latitutde  / rad

            # convert to degrees
            clon = np.rad2deg(clon_rad)
            clat = np.rad2deg(clat_rad)

            var = gdf[attribute]
            levels = np.linspace(np.nanmin(var), np.nanmax(var)+1e-9, contour_levels, endpoint=True) # add small margin to cope with rounding issues
            var = var.replace(np.nan, -999)

            if outlines is not None:
                outlines.boundary.plot(ax=ax, linewidth=1, edgecolor=outlines_color, alpha=0.5)
                ax.set(xlim=(clon.min(), clon.max()), ylim=(clat.min(), clat.max()))

            if v_lim is not None:
                levels = np.linspace(v_lim[0], v_lim[1], contour_levels, endpoint=True)
                tcf = ax.tricontourf(clon, clat, var, levels=levels, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1], extend="both")
            else:
                tcf = ax.tricontourf(clon, clat, var, levels=levels, cmap=cmap) #  cmap = plt.cm.gist_earth)

            if colorbar:
                plt.colorbar(tcf, ax=ax, shrink=0.5)

            if contour_lines:
                tc = plt.tricontour(clon, clat, var, levels=levels,
                               colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
                               linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])
                if contour_labels:
                    ax.clabel(tc, tc.levels[::5], inline=True, fontsize=10)

        ax.set_title(title, fontweight = 'bold')
        ax.grid(False)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_albedo_all_months(self, band='vis', cmap = 'jet', show=True, save_path=None):
        gdf = self.get_triangles()

        fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(10, 12), sharex=True, sharey=True)
        norm = Normalize(vmin=10, vmax=90)
        for month in range(1, 13):
            col = (month - 1) % 3
            row = int((month - 1) / 3)
            print(f'month {month}: {row}|{col}')
            ax = axs[row][col]
            gdf[band] = self.get_albedo(month, band)
            gdf.plot(band, ax=ax, cmap='jet',
                       norm=norm)  # legend=False, legend_kwds={"shrink": .7}, vmin=10, vmax=90)
            ax.set_title(f'Month {month}')

        # Create an axis for the colorbar
        cax = fig.add_axes([0.1, 0.04, 0.8, 0.02])

        # Add a colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        # cbar = fig.colorbar(sm, cax=cax)
        fig.colorbar(sm, cax=cax, orientation='horizontal')

        plt.suptitle('Albedo for Grid', fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        # plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def get_ICONDataGrid(self, data_file):
        return ICONDataGrid(data_file, self)

    def get_ICONInitGrid(self, data_file):
        return ICONInitGrid(data_file, self, self.ext_param_file)

    def get_ICONCell(self, cell_id):
        return ICONCell(self, cell_id)

###


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

###

class ICONInitGrid:

    def __init__(self, data_file, grid, ext_file):
        self.ds = xr.open_dataset(data_file, autoclose=True)
        self.ext = xr.open_dataset(ext_file, autoclose=True)
        self.grid = grid

    def add_snow_to_ice(self):
        print(self.ds.ncells)
        for i in self.ds.ncells:
            if self.ext['ICE'][i]>0.0:
                self.ds['H_SNOW'][0,i] = 1.0
                self.ds['RHO_SNOW'][0,i] = 900.0
                self.ds['W_SNOW'][0,i] = self.ds['H_SNOW'][0,i] * self.ds['RHO_SNOW'][0,i]
                self.ds['FRESHSNW'][0,i] = 0.9
                #self.ds_icon['SNOAG'][0,i] = 365
                self.ds['T_SNOW'][0,i] = 270.0
            else:
                self.ds['H_SNOW'][0,i] = 0.0
                self.ds['RHO_SNOW'][0,i] = 0.0
                self.ds['W_SNOW'][0,i] = self.ds['H_SNOW'][0,i] * self.ds['RHO_SNOW'][0,i]
                self.ds['FRESHSNW'][0,i] = 0.0
                #self.ds_icon['SNOAG'][0,i] = 365:
                self.ds['T_SNOW'][0,i] = self.ds['T_SNOW'][0,i]

    def write_init(self, fname):
        self.ds.to_netcdf(fname)
        print('done!')

###
class ICONCell:

    def __init__(self, grid, cell_id):
        # load .nc files
        self.grid = grid
        self.cell_id_idx = cell_id - 1

    def get_lu_class_fractions(self, ignore_zero=True):
        df = pd.DataFrame()
        df['lu_name'] = self.grid.get_lu_class_name()
        df['lu_fraction'] = self.grid.ds_ext['LU_CLASS_FRACTION'][:, self.cell_id_idx].data
        if ignore_zero:
            df = df.loc[df['lu_fraction'] > 0]
        return df

    def plot_lu_class_fractions(self, show=True, save_path=None):
        df = self.get_lu_class_fractions()
        print(df)
        fig, ax = plt.subplots()
        ax.pie(df['lu_fraction'], labels=df['lu_name'], autopct='%1.2f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('LU Fractions')
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()


###
class ICONMeteogram:

    def __init__(self, file_meteogram, file_icon_data=None):
        if file_icon_data is not None:
            self.ds_icon_data = xr.load_dataset(file_icon_data, autoclose=True)
        self.ds_meteogram = xr.load_dataset(file_meteogram, autoclose=True)

        # defaults
        self.cc_var_idx = 0
        self.met_station_idx = 0
        self.cell_id_idx = None

    def get_all_locations(self):
        df = pd.DataFrame()
        for i in range(1, self.ds_meteogram.nstations.size+1):
            info = pd.DataFrame([self.get_location_info(i)])
            df = pd.concat([df, info], ignore_index=True)
        return df

    def get_all_variables(self):
        df = pd.DataFrame()
        for i in range(1, self.ds_meteogram.nvars.size+1):
            info = pd.DataFrame([self.get_variable_info(i)])
            df = pd.concat([df, info], ignore_index=True)
        return df

    def get_location_info(self, meteogram_station):
        loc_idx = meteogram_station - 1
        location = {'id':meteogram_station,
                    'idx': loc_idx,
                    'name': self.ds_meteogram['station_name'].values[loc_idx].decode("utf-8"),
                    'lat': round(self.ds_meteogram['station_lat'].values[loc_idx], 4),
                    'lon': round(self.ds_meteogram['station_lon'].values[loc_idx], 4),
                    'alt': round(self.ds_meteogram['station_hsurf'].values[loc_idx])
                    }
        return location

    def get_variable_info(self, variable_id):
        var_idx = variable_id - 1
        variable = {'id': variable_id,
                    'idx': var_idx,
                    'name': self.ds_meteogram['var_name'].values[var_idx].decode("utf-8"),
                    'long_name': self.ds_meteogram['var_long_name'].values[var_idx].decode("utf-8"),
                    'unit': self.ds_meteogram['var_unit'].values[var_idx].decode("utf-8"),
                    'nlevs': self.ds_meteogram['var_nlevs'].values[var_idx],
                    'group_id': self.ds_meteogram['var_group_id'].values[var_idx]
                    }
        return variable

    def plot_meteogram(self, cell_id, meteogram_station, cc_var=20, save_path=None, overwrite_name=None,
                       from_dt=None, to_dt=None, figsize=(10, 8)):
        # prepare
        self.cell_id_idx = cell_id - 1
        self.met_station_idx = meteogram_station - 1
        self.cc_var_idx = cc_var - 1

        self.__load_values()

        if overwrite_name is None:
            name = self.location['name']
        else:
            name = overwrite_name
        lon = self.location['lon']
        lat = self.location['lat']
        alt = self.location['alt']

        # START OF PLOT
        fig, axs = plt.subplot_mosaic([['temp'], ['wind'], ['cc']],
                                      constrained_layout=True,
                                      figsize=figsize)

        axs['wind'].sharex(axs['cc'])
        axs['temp'].sharex(axs['wind'])

        # plot location/time data above first row
        ax = axs['temp']
        ax.set_title(f'Hintereisferner {name}: {lon}°, {lat}° ({alt}m)', loc='left', size=8)
        ax.set_title(f"from {self.min_dt.strftime('%d.%m.%y %H:%M')} to {self.max_dt.strftime('%d.%m.%y %H:%M')} UTC",
                     loc='right', size=8)

        # -> TEMP (1st row; left)
        self.__add_sunhours(ax)
        ax.set_ylim(self.aT.min().m - 2, self.aT.max().m + 2)
        ax.plot(self.icon_time, self.aT, color='#f30000', alpha=1, linestyle='-', linewidth='2', zorder=4)
        ax.fill_between(self.icon_time, self.aT, color='#f30000', alpha=0.1)
        ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='gray', linewidth=0.4)
        ax.tick_params(axis="x", direction="in", pad=-12, which='both')
        ax.xaxis.set_tick_params(labelbottom=False, which='both')
        ax.set_ylabel('Temperature (°C)', color='red')

        # -> PRESSURE (1st row; right)
        axr = ax.twinx()
        axr.plot(self.icon_time, self.p, color='green', alpha=1, linestyle='dotted', linewidth=2)
        axr.set_ylim(self.p.max().m + 2, self.p.min().m - 2)
        axr.invert_yaxis()
        axr.set_ylabel('Pressure (hPa)', color='green')

        # -> WIND (2nd row)
        ax = axs['wind']
        self.__add_sunhours(ax)
        # ax.set_title('WIND', loc='left')
        ax.set_ylim(0, self.wg.max().m + 6)
        ax.plot(self.icon_time, self.ws, color='#3f91ff', alpha=1, linewidth=2)
        ax.fill_between(self.icon_time, self.ws, color='#3f91ff', alpha=0.1)
        ax.plot(self.icon_time, self.wg, color='#379696', linestyle='-', alpha=1, linewidth=1.5)
        ax.barbs(self.icon_time[::3], self.wg.max().m + 3, self.u[::3], self.v[::3], length=5, alpha=0.7, pivot='middle',
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
        xs = mpl.dates.date2num(self.met_time)

        ax = axs['cc']
        self.__add_sunhours(ax)
        ax.imshow(self.cc, extent=[xs[0], xs[-1], self.var_cc['nlevs'], 0],
                  aspect='auto', cmap=cmap, alpha=1, zorder=9,
                  interpolation='nearest')
        ax.set_yticklabels(np.round(self.heights[::10], 1))
        ax.grid(which='major', axis='y', linestyle=(0, (5, 10)), color='gray', linewidth=0.4)
        ax.set_ylabel('Altitude (km)')

        # Format the x-axis
        min_dt = self.min_dt if from_dt is None else from_dt
        max_dt = self.max_dt if to_dt is None else to_dt
        ax.set_xlim(min_dt, max_dt)
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


    def __load_values(self):
        # Get time from ICON data .nc file (float formatted)
        time = np.array(self.ds_icon_data['time'][:].values)
        icon_time = [ICONDataGrid.convert_float_dt(t) for t in time]
        self.icon_time = pd.to_datetime(icon_time, utc=True)

        # Get variables from ICON data .nc file
        self.p = np.array(self.ds_icon_data['pres_sfc'][:, self.cell_id_idx].values) / 100 * units.hPa
        self.u = np.array(self.ds_icon_data['u_10m'][:, 0, self.cell_id_idx].values) * units('m/s')
        self.v = np.array(self.ds_icon_data['v_10m'][:, 0, self.cell_id_idx].values) * units('m/s')
        self.wg = np.array(self.ds_icon_data['gust10'][:, 0, self.cell_id_idx].values) * units('m/s')
        self.aT = (np.array(self.ds_icon_data['t_2m'][:, 0, self.cell_id_idx].values) - 273.15) * units.degC

        # Calculate wind speed and direction
        self.wd = mpcalc.wind_direction(self.u, self.v)
        self.ws = mpcalc.wind_speed(self.u, self.v)

        # Cloud Cover from Meteogram .nc file
        self.var_cc = self.get_variable_info(self.cc_var_idx+1)
        print(f'Please check used variable for Cloud Cover: {self.var_cc}')
        self.cc = np.array(self.ds_meteogram['values'][:, :self.var_cc['nlevs'], self.cc_var_idx, self.met_station_idx].values.T)

        # Get heights for cloud cover (in km)
        self.heights = self.ds_meteogram['heights'][:, self.cc_var_idx, self.met_station_idx].values / 1000

        # Get timesteps from meteogram (might be different from ICON data) and convert to datetime
        met_time = self.ds_meteogram['date'][:].values
        met_time = np.char.decode(met_time)
        self.met_time = pd.to_datetime(met_time, utc=True)

        # Overall Time frame
        self.min_dt = min(min(self.icon_time), min(self.met_time))
        self.max_dt = max(max(self.icon_time), max(self.met_time))
        print(f'Time frame: {self.min_dt} - {self.max_dt}')

        # selected location
        self.location = self.get_location_info(self.met_station_idx + 1)
        print('Location Data: ', self.location)

        # init SunTimes for location
        self.sun = SunTimes(longitude=self.location['lon'],
                            latitude=self.location['lat'],
                            altitude=self.location['alt'])

    def __add_sunhours(self, ax):
        previous_day = None

        # Iterate through the list of datetimes
        for current_datetime in self.icon_time:
            current_day = current_datetime.date()
            if current_day != previous_day:
                sunrise = self.sun.risewhere(current_day, 'UTC')
                sunset = self.sun.setwhere(current_day, 'UTC')

                # add to plot
                ax.axvspan(sunrise, sunset, facecolor='#f0f8c5', alpha=1)
                ax.axvline(current_day, linewidth=0.8, color='lightgrey')
                previous_day = current_day

