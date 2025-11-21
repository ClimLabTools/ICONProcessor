import xarray as xr

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
