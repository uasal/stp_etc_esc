# stp_etc_esc
This is intended to be the active directory for the public release of the exposure time calculator for the ExtraSolar Camera.

Derived from the exposure time calculator developed by Aaron Goldtooth, with updates from Justin Hom

Additional Contributions from Sanchit Sabhlok and Jess Johnson.

## Requirements
The package requires a few configuration repositories to be set up and installed in your python environment.
1. [config_um](https://github.com/uasal/config_um) - The configuration repo for Ultramarine 3 m telescope.
2. [config_stp](https://github.com/uasal/config_stp) - The configuration repo for the Space Telescope Project.
3. [config_stp_esc](https://github.com/uasal/config_stp_esc) - Configuration repo for the ESC instrument.

All important properties related to the instrument including mirrors, optics, coating throughput, filters, and detectors, are stored in these repositories. Note that the exposure time calculator also allows for users to input their own configurations, but this feature can only be used if the configuration(s) follow the same organizational structure as defined in these configuration repositories. Users can also use the class functions built into the backend to create their own instruments for calculating exposure times.

The installation instructions for these packages can be found on their corresponding github repos. 

## Dependencies
All UASAL config packages are dependent on [utils_config](https://github.com/uasal/utils_config) but will be automatically installed when installing the configuration repositories.

Additional package dependencies required for the ETC - astropy, numpy, matplotlib, scipy and synphot.

## Installation
The Exposure Time Calculator is a python package that can be installed via a download from github and then installing on your system locally. You can directly clone the [ETC github repo](https://github.com/uasal/stp_etc_esc) or fork the repo and clone the fork. 
```
$ git clone git@github.com:uasal/stp_etc_esc.git
$ cd stp_etc_esc
$ pip install .
```
The ETC should now be installed on your local machine. To confirm installation, the following import on python should work - 
```
import stp_etc_esc
stp_etc_esc.__version__
```

## Usage

Included in this repository is an [example notebook] of how an analysis would make use of the ESC exposure time calculator.
What is included in this readme is only a brief summary.



