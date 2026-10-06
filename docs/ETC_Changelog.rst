List of changes from the original ETC
=====================================

The ETC was originally created by forking the UASAL Exposure Time Calculator repo originally written by Aaron Goldtooth. The fork was first created in July 2024. This document lists the changes that have been made since the fork was made between the forked version and the initial ETC release.

1. Removed unused file data and matched directory structure with config repos.

2. Refactored `ExposureTimeSNRCalculator.py` to use the config structure.

3. Created independently from wcc_exposure_time_calculator

4. Added `pyproject.toml` to make repo installable. Changed directory structure to make package installable.

5. Added versioning to the package.

6. Added the functionality to import the template in the code.

7. Cleaned up existing notebooks, retaining only the demo notebook. Notebook modified to match the new formatting in code.

8. Added functionality to include a `support_data_path` which is the base of the directory where the file can be found. Standardized this across the ETC.

9. Removed all hard coded numbers and pointed them to the corresponding entry in the configuration toml files.

10. The code `calc_SNR`, `calc_int_time` and `calc_req_source` functions has been refactored for clarity. The returned units have also been checked and corrected.

11. The checks for the QE being passed to the `calc_SNR`, `calc_int_time` and `calc_req_source` functions have been removed. The implicit assumption is that the detector throughput curve has been provided to the ETC when the sensor is added.

12. Fixed read noise implementation (previously was not being squared) after confirming the source of e- read noise is actually RMS electrons per frame. Also fixed the corresponding unit check when calling `config_stp` respectively.

13. Sky background normalization now performed in the Johnson V band, as this is the source of Zodi background SB from HST.

14. All noise terms are now evaluated over one photometric aperture. The planet and speckle counts are measured through the vortex core-throughput curve (Ruane et al. 2018), i.e. inside r = 0.7 lambda/D, but the zodi/exozodi terms used ``resel = pi (lambda/2D)^2`` (half that area) and the dark/read terms ``num_psf_pixels = ceil(2.44 lambda f#/pixel)^2`` (4-5x that area). ``Observatory.set_photometric_aperture(model='core')`` (called by ``make_STP``; the default) sets ``resel = pi (0.7 lambda/D)^2`` and ``num_psf_pixels = resel / plate_scale^2`` at the science-filter centre; ``model='legacy'`` and ``calc_PSF(approx_type='sq'|'circ')`` restore the previous behaviour. Because the ESC is read-noise limited for faint hosts, the legacy apertures inflated integration times by ~3-4x. The hard-coded ``Npix = 225`` override in the EXOSIMS-style cross-check test was removed.
