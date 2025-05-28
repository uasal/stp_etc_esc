List of changes from the original ETC
=====================================

The ETC was originally created by forking the UASAL Exposure Time Calculator repo originally written by Aaron Goldtooth. The fork was first created on February 12th, 2025. This document lists the changes that have been made since the fork was made between the forked version and the initial ETC release.

1. Removed unused file data and matched directory structure with config repos.

2. Refactored `ExposureTimeSNRCalculator.py` to use the config structure. Added functions to interpolate sensor files. The inputs are no longer hard coded in code but filenames must be passed into functions to get the corresponding calibration.

3. Added `pyproject.toml` to make repo installable. Changed directory structure to make package installable.

4. Added versioning to the package.

5. Added the functionality to import the template in the code.

6. Cleaned up existing notebooks, retaining only the demo notebook. Notebook modified to match the new formatting in code.

7. Added functionality to include a `support_data_path` which is the base of the directory where the file can be found. Standardized this across the ETC.

8. Added a convenience function to calculate zero magnitude source counts.

9. Removed all hard coded numbers and pointed them to the corresponding entry in the configuration toml files.

10. Default bandpass changed to Sloan g band.

11. The code `calc_SNR`, `calc_int_time` and `calc_req_source` functions has been refactored for clarity. The returned units have also been checked and corrected.

12. The checks for the QE being passed to the `calc_SNR`, `calc_int_time` and `calc_req_source` functions have been removed. The implicit assumption is that the detector throughput curve has been provided to the ETC when the sensor is added.

13. Added explicit `surf_area` variable.

14. Changed default magnitudes to AB mag across the whole ETC.

15. Gain no longer an option that can be provided into SNR calculations, must be specified implicitly through sensor calibration files.

16. Fixed the bugs in `calc_req_source` and `calc_int_time` functions.

17. Fixed the SNR scaling in `calc_req_source` function, to now only take square root of the unit, not the value when calculating required source counts/magnitudes.

18. Added the option to use either "UM" or "STP" when calling `make_STP`. This initializes the telescope from either `config_um` or `config_stp` respectively.

19. Fixed read noise implementation (previously was not being squared) after confirming the source of e- read noise is actually RMS electrons per frame. Also fixed the corresponding unit check when calling `config_stp` respectively.

20. Sky background no longer calculated by scaling the PSF area with the detector area. Now calculated by estimating the total magnitude given the size of the PSF and a surface brightness. This can now be applied to use galaxies as an external background source as well.

21. Added a Jupyter example notebook to demonstrate SNR calculations for a supernova in an embedded galaxy.

22. Sky background normalization now performed in the Johnson V band, as this is the source of Zodi background SB from HST.

23. The filter is now added using the `from_file` function instead of the `from_filter` function.