"""Shared pieces of the historical UM regression (see README.md in this folder).

`make_um_baseline.py` and `check_um_regression.py` must run the ETC in exactly
the same way, so the configuration, the observation inputs and the list of
summary quantities live here, in one place.

This module never builds a wavelength grid: the grid is fixed reference data
stored in tests/test_data/bandpass_baseline_um.csv.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import astropy.units as u

from stp_etc_esc import ExposureTimeSNRCalculatorESC as etsc


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPO_ROOT / "tests" / "test_data"
CURVE_BASELINE = TEST_DATA / "bandpass_baseline_um.csv"
SUMMARY_BASELINE = TEST_DATA / "um_baseline_summary.csv"

BANDPASS_NAMES = ["bandpass", "precoron_bandpass", "coron_bandpass"]

# Supported Ultramarine configuration (default STP currently fails upstream
# with KeyError: 'dichroic', so it cannot provide a baseline).
MAKE_STP_ARGS = {"telconfig": "UM", "escconfig": "UM", "telpath": "UM", "escpath": "UM"}

# Same call pattern as tests/test_esc_etc_initialization.py::test_counts,
# with the host/zodi/exozodi magnitudes from the UM configuration.
GENERIC_SOURCE_ARGS = {"contrast": 1e-8, "starMag": 0.0}
OBSERVATION_ARGS = {"hoststarflux": 0.0, "planetdeltamag": 20, "bg_flux": 22.5,
                    "flux_units": "vega", "exobg_flux": 22}
SNR_INT_TIME = 600.0 * u.s
SNR_EXP_TIME = 10.0 * u.s
INT_TIME_SNR = 5
INT_TIME_EXP_TIME = 10.0 * u.s

# Names for the five values calc_int_time() returns, in the order it builds
# them (ExposureTimeSNRCalculatorESC.calc_int_time, `noise_terms = np.array([...])`).
NOISE_TERM_NAMES = [
    "dark_current_x_npix",               # (dark_current*num_psf_pix)
    "read_noise_sq_x_npix_per_exptime",  # ((readout_noise**2)*num_psf_pix/(exp_time))
    "sky_counts",                        # sky_counts (zodi)
    "exozodi_counts",                    # exozodi_counts
    "speckle_counts",                    # speckle_counts
]

PURPOSE_NOTE = ("Historical regression environment: reproduces the behaviour of the "
                "unmodified ETC. These package versions are NOT project dependency requirements.")


def progress(message):
    print(message, flush=True)


def build_um_observatory():
    """Build the observatory exactly as the baseline did (UM config, make_STP only)."""
    obs = etsc.Observatory("UM", 2.4 * u.m, 36.45 * u.m)
    obs.make_STP(**MAKE_STP_ARGS)
    return obs


def evaluate_bandpasses(obs, wavelengths_angstrom):
    """Return {bandpass name: throughput array} at the given wavelengths (Angstrom)."""
    wavelengths = np.asarray(wavelengths_angstrom, dtype=float) * u.AA
    values = {}
    for name in BANDPASS_NAMES:
        bandpass = getattr(obs, name)
        values[name] = np.asarray(bandpass(wavelengths).value, dtype=float)
    return values


def _row(name, quantity_or_number):
    """One summary row: (quantity name, numeric value, unit string)."""
    if isinstance(quantity_or_number, u.Quantity):
        value = float(np.squeeze(quantity_or_number.value))
        unit = quantity_or_number.unit.to_string() or "dimensionless"
    else:
        value = float(quantity_or_number)
        unit = "dimensionless"
    return (name, value, unit)


def compute_summary(obs):
    """All scalar regression quantities, in a fixed order.

    This runs the expensive synphot integrals (~2 min) and make_observation
    (~5 min) for the UM configuration. It modifies `obs` (source, background,
    count rates) the same way a normal ETC user would.
    """
    rows = []

    progress("  bandpass peak / width integrals ...")
    for name in BANDPASS_NAMES:
        bandpass = getattr(obs, name)
        rows.append(_row(f"{name}.wpeak", bandpass.wpeak()))
        rows.append(_row(f"{name}.tpeak", bandpass.tpeak()))
        rows.append(_row(f"{name}.rectwidth", bandpass.rectwidth()))

    rows.append(_row("num_mirrors", obs.num_mirrors))
    rows.append(_row("len_qe_curves", len(obs.qe_curves)))
    # Each qe_curves entry is "<what> x <num_curves>". <what> is an absolute
    # file path for file-based components, but a plain value for the others
    # (generic filter wavelength, generic optic throughput, FPM lambda/D).
    # Only file paths are shortened, because they are machine specific.
    for index, entry in enumerate(obs.qe_curves, start=1):
        what, count = entry.rsplit(" x ", 1)
        if os.path.isabs(what):
            what = os.path.basename(what)
        rows.append((f"qe_curves[{index:02d}]", f"{what} x {count}", "text"))

    rows.append(_row("plate_scale", obs.plate_scale))
    rows.append(_row("num_psf_pixels", obs.num_psf_pixels))
    rows.append(_row("qe_wpeak", obs.qe_wpeak))
    rows.append(_row("surf_area", obs.surf_area))

    progress("  make_observation (slow for UM: several minutes) ...")
    obs.set_generic_source(**GENERIC_SOURCE_ARGS)
    obs.set_background(background_file=None)
    obs.make_observation(**OBSERVATION_ARGS)

    # hoststar_counts is intentionally absent: make_observation never sets it.
    for name in ["source_counts", "sky_counts", "speckle_counts", "exozodi_counts"]:
        rows.append(_row(name, getattr(obs, name)))

    rows.append(_row("calc_SNR(600 s, 10 s)", obs.calc_SNR(SNR_INT_TIME, SNR_EXP_TIME)))
    int_time, noise_terms = obs.calc_int_time(INT_TIME_SNR, INT_TIME_EXP_TIME)
    rows.append(_row("calc_int_time(5, 10 s)", int_time))
    for name, value in zip(NOISE_TERM_NAMES, noise_terms):
        rows.append(_row(f"calc_int_time.noise_terms.{name}", value))
    rows.append(_row("calc_saturation_time", obs.calc_saturation_time()))
    return rows


def _installed_commit(package):
    direct_url = distribution(package).read_text("direct_url.json")
    if not direct_url:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


def _git(*args):
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def etc_source_is_dirty():
    return _git("status", "--porcelain", "--", "src") != ""


def provenance():
    """Provenance as one dict, so every key appears exactly once."""
    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stp_etc_esc_commit": _git("rev-parse", "HEAD"),
        "stp_etc_esc_src_dirty": str(etc_source_is_dirty()),
        "config_stp_commit": _installed_commit("config_stp"),
        "config_stp_esc_commit": _installed_commit("config_stp_esc"),
        "config_um_commit": _installed_commit("config_um"),
        "config_um_esc_commit": _installed_commit("config_um_esc"),
        "utils_config_commit": _installed_commit("utils_config"),
        "python": sys.version.split()[0],
    }
    for package in ["numpy", "astropy", "synphot", "scipy", "pandas"]:
        info[package] = distribution(package).version
    info["observatory"] = "Observatory('UM', 2.4*u.m, 36.45*u.m)"
    info["make_STP_args"] = str(MAKE_STP_ARGS)
    info["set_generic_source_args"] = str(GENERIC_SOURCE_ARGS)
    info["set_background_args"] = "{'background_file': None}"
    info["make_observation_args"] = str(OBSERVATION_ARGS)
    info["calc_SNR_args"] = f"({SNR_INT_TIME}, {SNR_EXP_TIME})"
    info["calc_int_time_args"] = f"({INT_TIME_SNR}, {INT_TIME_EXP_TIME})"
    return info


def metadata_lines():
    """'# '-comment header lines written at the top of both baseline files."""
    return [PURPOSE_NOTE] + [f"{key}: {value}" for key, value in provenance().items()]


def read_metadata(path):
    """Return the '# key: value' header lines of a baseline file as a dict."""
    metadata = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if ": " in text:
                key, value = text.split(": ", 1)
                metadata[key] = value
    return metadata
