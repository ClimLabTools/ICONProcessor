from .ICONDataGrid import ICONDataGrid
from .ICONInitGrid import ICONInitGrid
from .ICONCell import ICONCell

import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from shapely.geometry import Polygon, Point
from matplotlib.colors import Normalize, ListedColormap
from colorama import Fore


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
                    ax=None, colorbar=True, v_lim=None, extend='neither'):

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
                tcf = ax.tricontourf(clon, clat, var, levels=levels, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1], extend=extend)
            else:
                tcf = ax.tricontourf(clon, clat, var, levels=levels, cmap=cmap, extend=extend)

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