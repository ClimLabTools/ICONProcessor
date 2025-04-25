from setuptools import setup, find_packages

setup(
    name="ICONProcessor",                  # name of your package
    version="0.1",
    packages=find_packages(),             # auto-detect packages
    license='MIT',
    python_requires='>=3.11',
    install_requires=[
        "geopandas==1.0.1",
        "netcdf4==1.7.2",
        "metpy==1.6.3",
        "colorama==0.4.6",
        "suntimes==1.1.2"
    ],                  # add dependencies here if needed
    description="Utility functions to process ICON grid data and model output data",
    author="Alexander Georgi",
    url="https://github.com/ClimLabTools/ICONProcessor",  # link to repo
)
