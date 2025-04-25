from setuptools import setup, find_packages

setup(
    name="ICONProcessor",                  # name of your package
    version="0.1",
    packages=find_packages(),             # auto-detect packages
    install_requires=[],                  # add dependencies here if needed
    description="Utility functions to process ICON grid data and model output data",
    author="Alexander Georgi",
    url="https://github.com/ClimLabTools/ICONProcessor",  # link to repo
)
