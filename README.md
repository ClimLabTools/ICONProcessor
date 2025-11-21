# ICONProcessor
Python tool kit for ICON (Icosahedral Nonhydrostatic) Model.

The library consists of multiple classes:
- `ICONGrid`: for time and variable independent grid information and external parameters
- `ICONInitGrid`: for initial conditions
- `ICONDataGrid`: for actual model outputs (full grid or data points)
- `ICONCell`: for single cell
- `ICONMeteogram`: for meteogram output files

Refer to the notebooks in the `Examples` folder to learn how to use the library.

---

How to install `ICONProcessor` into Python environment:

```
pip install git+https://github.com/ClimLabTools/ICONProcessor.git
```

---

How to create your own kernel for Jupyterlab incl. the `ICONProcessor` (e.g. at Levante):

```
conda create -n icon_proc -c conda-forge ipykernel python=3.11
source activate icon_proc
pip install git+https://github.com/ClimLabTools/ICONProcessor.git
python -m ipykernel install --user --name icon_proc_kernel --display-name="ICONProcessor"
```

Find more details at the DKRZ help page:
https://docs.dkrz.de/doc/software&services/jupyterhub/kernels.html