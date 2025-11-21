import matplotlib.pyplot as plt
import pandas as pd


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