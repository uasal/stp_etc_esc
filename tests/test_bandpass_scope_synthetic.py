"""Freeze how each add_*() method currently changes the three ETC bandpasses.

These tests describe the EXISTING (pre-refactor) behaviour of
ExposureTimeSNRCalculatorESC, so that the component-throughput-budget
refactor can prove it keeps that behaviour. They use tiny synthetic curves
written to a temporary directory, so they do not depend on the STP or UM
configuration data and run in any synphot version.

For each component they check its value in:
    bandpass           - full throughput, used for count rates
    precoron_bandpass  - throughput without the coronagraph FPM and
                         detector QE
    coron_bandpass     - coronagraph-only throughput

Not covered here: add_filter() (needs synphot's reference filter data and
a 'fixed_filters' directory; make_STP() never calls it) and plotting.
The real UM configuration is covered by tests/regression/ instead.
"""

import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits

from stp_etc_esc import ExposureTimeSNRCalculatorESC as etsc


# A new Observatory starts all three bandpasses as a box of 1 from 4000 to
# 14000 Angstrom, so test wavelengths inside that range see only the
# component(s) under test.
INSIDE = 6000.0      # Angstrom
OUTSIDE = 3000.0     # Angstrom, below the starting box


def new_observatory():
    return etsc.Observatory("synthetic", 3 * u.m, 36 * u.m)


def throughputs(obs, wavelength_angstrom):
    """(bandpass, precoron_bandpass, coron_bandpass) at one wavelength."""
    wavelength = wavelength_angstrom * u.AA
    return (float(obs.bandpass(wavelength).value),
            float(obs.precoron_bandpass(wavelength).value),
            float(obs.coron_bandpass(wavelength).value))


def write_flat_curve(path, value):
    """Two-column CSV curve (wavelength in nm, throughput), flat at `value`."""
    lines = ["Wavelength nm,Throughput"]
    for wavelength_nm in range(200, 1600, 100):
        lines.append(f"{wavelength_nm},{value}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_two_column_csv(path, header, rows):
    """CSV in the format add_sensor() reads: one header line, then x,y rows."""
    lines = [header] + [f"{x},{y}" for x, y in rows]
    path.write_text("\n".join(lines) + "\n")


def write_fpm_table(path):
    """FPM throughput table in the format add_lamD_optic() reads:
    row 0 = separation in lambda/D, row 1 = throughput in percent."""
    table = np.array([[1.0, 2.0, 3.0, 4.0],
                      [10.0, 20.0, 30.0, 40.0]])
    fits.writeto(path, table)
    return str(path)


def separation_arcsec(obs, lambda_over_d):
    """Separation in arcsec of `lambda_over_d` at the filter wavelength."""
    one_lambda_over_d = (obs.primary_filter / obs.diameter_primary).to(
        u.arcsec, equivalencies=u.dimensionless_angles())
    return lambda_over_d * one_lambda_over_d.value


def test_new_observatory_starts_with_unity_box():
    obs = new_observatory()
    assert throughputs(obs, INSIDE) == pytest.approx((1.0, 1.0, 1.0))
    assert throughputs(obs, OUTSIDE) == pytest.approx((0.0, 0.0, 0.0))
    assert throughputs(obs, 15000.0) == pytest.approx((0.0, 0.0, 0.0))
    assert obs.qe_curves == []
    assert obs.num_mirrors == 0


def test_add_mirror(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=curve)
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 1.0))
    assert throughputs(obs, OUTSIDE) == pytest.approx((0.0, 0.0, 0.0))
    assert obs.num_mirrors == 1
    assert obs.qe_curves == [f"{curve} x 1"]


def test_add_mirror_num_curves_2(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=curve, num_curves=2)
    assert throughputs(obs, INSIDE) == pytest.approx((0.81, 0.81, 1.0))
    assert obs.num_mirrors == 2
    # One qe_curves entry per call.
    assert obs.qe_curves == [f"{curve} x 2"]


def test_add_qe_curve(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "qe.csv", 0.9)
    obs.add_qe_curve(curve, wave_unit="nm")
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 1.0))
    assert obs.qe_curves == [f"{curve} x 1"]


def test_add_transmissive_optic(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "polarizer.csv", 0.9)
    obs.add_transmissive_optic(curve, wave_unit="nm")
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 1.0))
    assert obs.qe_curves == [f"{curve} x 1"]


def test_add_transmissive_optic_coron_only_also_applies_everywhere(tmp_path):
    # coronOnly=True ADDS the optic to coron_bandpass; it is still applied
    # to bandpass and precoron_bandpass as well.
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "polarizer.csv", 0.9)
    obs.add_transmissive_optic(curve, wave_unit="nm", coronOnly=True)
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 0.9))


def test_add_generic_optic():
    obs = new_observatory()
    obs.add_generic_optic(0.9)
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 1.0))
    assert throughputs(obs, OUTSIDE) == pytest.approx((0.0, 0.0, 0.0))
    assert obs.qe_curves == ["0.9 x 1"]


def test_add_generic_optic_coron_only_also_applies_everywhere():
    obs = new_observatory()
    obs.add_generic_optic(0.9, coronOnly=True)
    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 0.9, 0.9))


def test_add_generic_optic_num_curves_2():
    obs = new_observatory()
    obs.add_generic_optic(0.9, num_curves=2)
    assert throughputs(obs, INSIDE) == pytest.approx((0.81, 0.81, 1.0))
    assert obs.qe_curves == ["0.9 x 2"]


def test_add_generic_filter():
    # 6300 Angstrom with 2% fractional bandwidth
    # -> 6237 to 6363 Angstrom, 0.98 inside.
    obs = new_observatory()
    obs.add_generic_filter(6300 * u.AA, fbw=0.02)
    for inside in [6240.0, 6300.0, 6360.0]:
        assert throughputs(obs, inside) == pytest.approx((0.98, 0.98, 1.0))
    for outside in [6230.0, 6370.0]:
        assert throughputs(obs, outside) == pytest.approx((0.0, 0.0, 1.0))
    assert obs.qe_curves == ["6300.0 Angstrom x 1"]


def test_add_generic_filter_num_curves_2_keeps_single_098():
    # Existing behaviour: extra copies use amplitude 1, not 0.98.
    obs = new_observatory()
    obs.add_generic_filter(6300 * u.AA, fbw=0.02, num_curves=2)
    assert throughputs(obs, 6300.0) == pytest.approx((0.98, 0.98, 1.0))
    assert obs.qe_curves == ["6300.0 Angstrom x 2"]


def test_add_lamD_optic_not_in_precoron(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0))
    # Applied to bandpass only: never to precoron_bandpass, and to
    # coron_bandpass only when coronOnly=True.
    assert throughputs(obs, INSIDE) == pytest.approx((0.2, 1.0, 1.0))
    entry_value, entry_count = obs.qe_curves[0].split(" x ")
    # The entry records lambda/D, not the file name.
    assert float(entry_value) == pytest.approx(2.0)
    assert entry_count == "1"


def test_add_lamD_optic_coron_only(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True)
    assert throughputs(obs, INSIDE) == pytest.approx((0.2, 1.0, 0.2))


def test_add_lamD_optic_uses_nearest_table_sample(tmp_path):
    # 2.4 lambda/D takes the 2 lambda/D sample (20%);
    # it is not interpolated (24%).
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.4))
    assert throughputs(obs, INSIDE)[0] == pytest.approx(0.2)


def test_add_sensor_qe_only_in_full_bandpass(tmp_path):
    # add_sensor() reads its noise curves through instrument_config, so
    # give it a minimal synthetic one. Only the QE's effect on the
    # bandpasses is checked.
    write_two_column_csv(tmp_path / "gain.csv", "gain_setting,gain",
                         [(0, 1.0), (200, 1.0)])
    write_two_column_csv(tmp_path / "dark.csv", "temperature,dark",
                         [(-30, 0.01), (30, 0.01)])
    write_two_column_csv(tmp_path / "read_noise.csv",
                         "gain_setting,read_noise",
                         [(0, 2.0), (200, 2.0)])
    write_two_column_csv(tmp_path / "well.csv", "gain_setting,well_depth",
                         [(0, 1e4), (200, 1e4)])
    qe_curve = write_flat_curve(tmp_path / "qe.csv", 0.9)

    obs = new_observatory()
    obs.instrument_SP = tmp_path
    obs.instrument_config = {
        "common_params": {
            "arm_a": {
                "sensor": {
                    "pixel_size": "4.63e-6m",
                    "gain_curve": "gain.csv",
                    "dark_current": "dark.csv",
                    "read_noise": "read_noise.csv",
                    "well_depth": "well.csv",
                }
            }
        }
    }
    obs.add_sensor(qe_curve, gain_setting=100, sensor_temp=0 * u.Celsius)

    assert throughputs(obs, INSIDE) == pytest.approx((0.9, 1.0, 1.0))
    assert obs.qe_curves == [f"{qe_curve} x 1"]


def test_components_multiply_cumulatively(tmp_path):
    # Each added component multiplies into the running total:
    # 0.9 x 0.8 x 0.5 = 0.36. Multiplication is commutative, so this does
    # not test ordering; registry order is tested once the registry exists.
    obs = new_observatory()
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=mirror)
    obs.add_generic_optic(0.8)
    obs.add_generic_optic(0.5)
    assert throughputs(obs, INSIDE) == pytest.approx((0.36, 0.36, 1.0))
    assert len(obs.qe_curves) == 3


def test_fpm_and_mirror_go_to_different_bandpasses(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=mirror)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True)
    assert throughputs(obs, INSIDE) == pytest.approx((0.18, 0.9, 0.2))


# ---------------------------------------------------------------------------
# Stage 1: the throughput component registry.
#
# These tests check that every add_*() call records the exact SpectralElement
# it applied, and which of the three bandpasses it applied it to, without
# changing how the bandpasses themselves are calculated.
# ---------------------------------------------------------------------------

def without_element(component):
    """A registry entry with the SpectralElement left out, for comparison."""
    return {key: value for key, value in component.items() if key != "element"}


def element_value(component, wavelength_angstrom=INSIDE):
    return float(component["element"](wavelength_angstrom * u.AA).value)


def add_synthetic_sensor(obs, directory, qe_value=0.9, name=None, group=None):
    """Call add_sensor() with minimal synthetic noise curves and a flat QE."""
    write_two_column_csv(directory / "gain.csv", "gain_setting,gain",
                         [(0, 1.0), (200, 1.0)])
    write_two_column_csv(directory / "dark.csv", "temperature,dark",
                         [(-30, 0.01), (30, 0.01)])
    write_two_column_csv(directory / "read_noise.csv",
                         "gain_setting,read_noise",
                         [(0, 2.0), (200, 2.0)])
    write_two_column_csv(directory / "well.csv", "gain_setting,well_depth",
                         [(0, 1e4), (200, 1e4)])
    qe_curve = write_flat_curve(directory / "qe.csv", qe_value)
    obs.instrument_SP = directory
    obs.instrument_config = {
        "common_params": {
            "arm_a": {
                "sensor": {
                    "pixel_size": "4.63e-6m",
                    "gain_curve": "gain.csv",
                    "dark_current": "dark.csv",
                    "read_noise": "read_noise.csv",
                    "well_depth": "well.csv",
                }
            }
        }
    }
    obs.add_sensor(qe_curve, gain_setting=100, sensor_temp=0 * u.Celsius,
                   name=name, group=group)
    return qe_curve


def test_registry_starts_empty():
    obs = new_observatory()
    assert obs.throughput_components == []


def test_registry_records_explicit_name_and_group(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=curve, name="m1", group="telescope")
    component = obs.throughput_components[0]
    assert component["name"] == "m1"
    assert component["group"] == "telescope"


def test_registry_add_qe_curve(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "qe.csv", 0.9)
    obs.add_qe_curve(curve, wave_unit="nm")
    assert len(obs.throughput_components) == 1
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "qe_curve", "category": "qe_curve", "group": None,
        "source": curve, "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.9)


def test_registry_add_sensor(tmp_path):
    obs = new_observatory()
    qe_curve = add_synthetic_sensor(obs, tmp_path)
    assert len(obs.throughput_components) == 1
    component = obs.throughput_components[0]
    # Detector QE is applied to the full bandpass only.
    assert without_element(component) == {
        "name": "sensor_qe", "category": "sensor_qe", "group": None,
        "source": qe_curve, "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": False,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.9)


def test_registry_add_mirror(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    obs.add_mirror(coating_file=curve, num_curves=2)
    assert len(obs.throughput_components) == 1
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "mirror", "category": "mirror", "group": None,
        "source": curve, "count": 2,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    # The element is the combined effect of both copies.
    assert element_value(component) == pytest.approx(0.81)


def test_registry_add_transmissive_optic(tmp_path):
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "polarizer.csv", 0.9)
    obs.add_transmissive_optic(curve, wave_unit="nm")
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "transmissive_optic", "category": "transmissive_optic",
        "group": None, "source": curve, "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.9)


def test_registry_add_transmissive_optic_coron_only(tmp_path):
    # coronOnly=True means "also apply to coron_bandpass".
    obs = new_observatory()
    curve = write_flat_curve(tmp_path / "polarizer.csv", 0.9)
    obs.add_transmissive_optic(curve, wave_unit="nm", coronOnly=True)
    component = obs.throughput_components[0]
    assert component["apply_to_bandpass"] is True
    assert component["apply_to_precoron"] is True
    assert component["apply_to_coron"] is True


def test_registry_add_generic_filter():
    obs = new_observatory()
    obs.add_generic_filter(6300 * u.AA, fbw=0.02)
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "generic_filter", "category": "generic_filter",
        "group": None, "source": 6300 * u.AA, "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    assert element_value(component, 6300.0) == pytest.approx(0.98)
    assert element_value(component, 6000.0) == pytest.approx(0.0)


def test_registry_add_generic_filter_num_curves_2_keeps_single_098():
    # The recorded element keeps the existing quirk: extra copies use
    # amplitude 1, so the filter is still 0.98, not 0.98 squared.
    obs = new_observatory()
    obs.add_generic_filter(6300 * u.AA, fbw=0.02, num_curves=2)
    component = obs.throughput_components[0]
    assert component["count"] == 2
    assert element_value(component, 6300.0) == pytest.approx(0.98)


def test_registry_add_generic_optic():
    obs = new_observatory()
    obs.add_generic_optic(0.9)
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "generic_optic", "category": "generic_optic",
        "group": None, "source": 0.9, "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.9)


def test_registry_add_generic_optic_coron_only():
    obs = new_observatory()
    obs.add_generic_optic(0.9, coronOnly=True)
    component = obs.throughput_components[0]
    assert component["apply_to_bandpass"] is True
    assert component["apply_to_precoron"] is True
    assert component["apply_to_coron"] is True


def test_registry_add_lamD_optic(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0))
    component = obs.throughput_components[0]
    assert component["name"] == "lamD_optic"
    assert component["category"] == "lamD_optic"
    assert component["group"] is None
    assert component["source"] == table
    assert component["count"] == 1
    # Never applied to precoron_bandpass.
    assert component["apply_to_bandpass"] is True
    assert component["apply_to_precoron"] is False
    assert component["apply_to_coron"] is False
    assert element_value(component) == pytest.approx(0.2)


def test_registry_add_lamD_optic_coron_only(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True)
    component = obs.throughput_components[0]
    assert component["apply_to_bandpass"] is True
    assert component["apply_to_precoron"] is False
    assert component["apply_to_coron"] is True


def test_registry_fpm_metadata_nearest_sample(tmp_path):
    # 2.4 lambda/D is recorded as requested, but the throughput used is
    # the nearest table sample (2 lambda/D -> 20%), not interpolated (24%).
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    table = write_fpm_table(tmp_path / "fpm.fits")
    separation = separation_arcsec(obs, 2.4)
    obs.add_lamD_optic(table, separation)
    component = obs.throughput_components[0]
    assert component["separation_arcsec"] == separation
    assert component["lambda_over_d"] == pytest.approx(2.4)
    assert component["selected_throughput"] == pytest.approx(0.2)
    assert element_value(component) == pytest.approx(0.2)


def test_registry_add_filter_both_branches(tmp_path, monkeypatch):
    from astropy.table import Table
    from synphot import SpectralElement
    from synphot.models import Box1D

    # Branch 1: a name containing '.fit' is read from a 'fixed_filters'
    # folder in the current directory and applied to the full bandpass only.
    filter_dir = tmp_path / "fixed_filters"
    filter_dir.mkdir()
    wavelengths = np.arange(2000.0, 15001.0, 1000.0) * u.AA
    Table({"WAVELENGTH": wavelengths,
           "THROUGHPUT": np.full(len(wavelengths), 0.8)}).write(
        filter_dir / "flat_filter.fits", format="fits")
    monkeypatch.chdir(tmp_path)

    obs = new_observatory()
    obs.add_filter("flat_filter.fits")
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "filter", "category": "filter", "group": None,
        "source": "flat_filter.fits", "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": False,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.8)
    assert throughputs(obs, INSIDE) == pytest.approx((0.8, 1.0, 1.0))
    # add_filter() writes to self.filters, not self.qe_curves.
    assert obs.filters == ["flat_filter.fits x 1"]
    assert obs.qe_curves == []

    # Branch 2: any other name is looked up with SpectralElement.from_filter,
    # which needs synphot reference data. Replace it with a flat 0.7 curve.
    def fake_from_filter(filter_name):
        return SpectralElement(Box1D, amplitude=0.7, x_0=20000, width=39999)

    monkeypatch.setattr(SpectralElement, "from_filter", fake_from_filter)

    obs = new_observatory()
    obs.add_filter("johnson_v")
    component = obs.throughput_components[0]
    assert without_element(component) == {
        "name": "filter", "category": "filter", "group": None,
        "source": "johnson_v", "count": 1,
        "apply_to_bandpass": True, "apply_to_precoron": True,
        "apply_to_coron": False,
    }
    assert element_value(component) == pytest.approx(0.7)
    assert throughputs(obs, INSIDE) == pytest.approx((0.7, 0.7, 1.0))
    assert obs.filters == ["johnson_v x 1"]


def test_registry_registration_order(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_mirror(coating_file=mirror, name="first")
    obs.add_generic_filter(6300 * u.AA, fbw=0.02, name="second")
    obs.add_generic_optic(0.5, name="third")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), name="fourth")
    names = [component["name"] for component in obs.throughput_components]
    assert names == ["first", "second", "third", "fourth"]
    # One registry entry per qe_curves entry, in the same order.
    assert len(obs.throughput_components) == len(obs.qe_curves)


def test_registry_reproduces_current_bandpasses(tmp_path):
    # Multiplying the starting box by every registered element whose flag
    # is True gives exactly the three bandpasses the ETC built.
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    polarizer = write_flat_curve(tmp_path / "polarizer.csv", 0.8)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_mirror(coating_file=mirror)
    obs.add_generic_filter(6300 * u.AA, fbw=0.02)
    obs.add_transmissive_optic(polarizer, wave_unit="nm", coronOnly=True)
    obs.add_generic_optic(0.5)
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True)
    add_synthetic_sensor(obs, tmp_path, qe_value=0.7)

    start = new_observatory()   # provides the same starting box
    rebuilt = {"bandpass": start.bandpass,
               "precoron_bandpass": start.precoron_bandpass,
               "coron_bandpass": start.coron_bandpass}
    for component in obs.throughput_components:
        if component["apply_to_bandpass"]:
            rebuilt["bandpass"] = rebuilt["bandpass"] * component["element"]
        if component["apply_to_precoron"]:
            rebuilt["precoron_bandpass"] = (rebuilt["precoron_bandpass"]
                                            * component["element"])
        if component["apply_to_coron"]:
            rebuilt["coron_bandpass"] = (rebuilt["coron_bandpass"]
                                         * component["element"])

    for wavelength in [3000.0, 6000.0, 6240.0, 6300.0, 6360.0, 13000.0]:
        for name, curve in rebuilt.items():
            expected = getattr(obs, name)(wavelength * u.AA).value
            assert curve(wavelength * u.AA).value == pytest.approx(
                expected, rel=1e-12, abs=0.0)


def write_synthetic_make_stp_inputs(directory):
    """Minimal custom telescope/instrument configs that make_STP() accepts.

    Every value make_STP() reads is present; the curves are flat files
    written to `directory`. Returns (telescope_config, instrument_config).
    """
    write_flat_curve(directory / "mirror.csv", 0.9)
    write_flat_curve(directory / "polarizer.csv", 0.8)
    write_flat_curve(directory / "qe.csv", 0.7)
    write_fpm_table(directory / "fpm.fits")
    write_two_column_csv(directory / "gain.csv", "gain_setting,gain",
                         [(0, 1.0), (200, 1.0)])
    write_two_column_csv(directory / "dark.csv", "temperature,dark",
                         [(-30, 0.01), (30, 0.01)])
    write_two_column_csv(directory / "read_noise.csv",
                         "gain_setting,read_noise",
                         [(0, 2.0), (200, 2.0)])
    write_two_column_csv(directory / "well.csv", "gain_setting,well_depth",
                         [(0, 1e4), (200, 1e4)])

    telescope_config = {
        "telescope": {
            "optics": {
                "m1": {"coating_refl": "mirror.csv", "surface_rms": "20e-9m"},
                "m2": {"coating_refl": "mirror.csv", "aper_clear_OD": "0.2m",
                       "support_width": "0.0m", "n_supports": 0},
            }
        },
        "observatory": {"pointing": {"jitter_rms": "1e-3arcsecond"}},
    }
    instrument_config = {
        "common_params": {
            "sources": {"companion": {"separation": 0.0866}},
            "ETC": {"pp_gain": 10.0, "rawDH_contrast_spec": 1e-8},
            "arm_a": {
                "general": {"f_number_sci_cam": 36.0},
                "pupil": {"aper_clear_OD": "3.0m"},
                # make_STP() walks these keys in this order.
                "optics": {
                    "oap1": {"coating_refl": "mirror.csv"},
                    "dichroic": {"dichroic_throughput": 0.9},
                    "lp1": {"pol_throughput": "polarizer.csv"},
                    "qwp1": {"pol_throughput": "none"},
                    "fpm": {"throughput": "fpm.fits"},
                    "lyot_stop": {"lyot_ratio": 0.9},
                    "filter": {"lam_central": "630.0nm", "bandwidth": 0.02},
                    "dm": {"coating_refl": "mirror.csv"},
                },
                "sensor": {
                    "qe": "qe.csv",
                    "gain": 100,
                    "temp_nominal": 0,
                    "pixel_size": "4.63e-6m",
                    "gain_curve": "gain.csv",
                    "dark_current": "dark.csv",
                    "read_noise": "read_noise.csv",
                    "well_depth": "well.csv",
                },
            },
        }
    }
    return telescope_config, instrument_config


EXPECTED_SYNTHETIC_COMPONENTS = [
    # (name, group, category, bandpass, precoron, coron)
    ("m1", "telescope", "mirror", True, True, False),
    ("m2", "telescope", "mirror", True, True, False),
    ("oap1", "instrument", "mirror", True, True, False),
    ("dm", "instrument", "mirror", True, True, False),
    ("filter", "instrument", "generic_filter", True, True, False),
    ("dichroic", "instrument", "generic_optic", True, True, False),
    ("lp1", "instrument", "transmissive_optic", True, True, False),
    ("lp1_pol_loss", "instrument", "generic_optic", True, True, False),
    ("qwp1", "instrument", "generic_optic", True, True, False),
    ("fpm", "instrument", "lamD_optic", True, False, True),
    ("detector_qe", "detector", "sensor_qe", True, False, False),
]


def test_make_stp_registry_names_groups_and_order(tmp_path):
    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs = new_observatory()
    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    recorded = [(c["name"], c["group"], c["category"], c["apply_to_bandpass"],
                 c["apply_to_precoron"], c["apply_to_coron"])
                for c in obs.throughput_components]
    assert recorded == EXPECTED_SYNTHETIC_COMPONENTS
    assert len(obs.throughput_components) == len(obs.qe_curves) == 11


def test_make_stp_resets_registry_but_not_num_mirrors(tmp_path):
    # The inputs are written once and used for both make_STP() calls.
    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs = new_observatory()

    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    assert len(obs.throughput_components) == 11
    assert obs.num_mirrors == 4

    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    # The registry and qe_curves start again ...
    assert len(obs.throughput_components) == 11
    assert len(obs.qe_curves) == 11
    # ... but num_mirrors keeps counting (existing behaviour, not changed).
    assert obs.num_mirrors == 8


# ---------------------------------------------------------------------------
# Stage 2 (written before the source change): make_STP() bandpass values.
#
# With the synthetic inputs every curve is flat, so the expected values can
# be written down by hand: four 0.9 mirrors, the 0.98 filter, the 0.9
# dichroic, the 0.8 polarizer, the 0.5 polarization loss, the 0.99 wave
# plate, the 0.2 FPM sample (about 2 lambda/D) and the 0.7 detector QE.
# ---------------------------------------------------------------------------

def test_make_stp_synthetic_bandpass_values(tmp_path):
    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs = new_observatory()
    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)

    optics = 0.9**4 * 0.98 * 0.9 * 0.8 * 0.5 * 0.99
    expected_bandpass = optics * 0.2 * 0.7      # plus FPM and detector QE
    expected_precoron = optics                  # no FPM, no detector QE

    def value(curve, wavelength_angstrom):
        return float(curve(wavelength_angstrom * u.AA).value)

    assert value(obs.bandpass, 6300.0) == pytest.approx(
        expected_bandpass, rel=1e-12, abs=0.0)
    assert value(obs.precoron_bandpass, 6300.0) == pytest.approx(
        expected_precoron, rel=1e-12, abs=0.0)
    # coron_bandpass is the make_STP() starting box (1500-16500 Angstrom)
    # times the FPM.
    assert value(obs.coron_bandpass, 6300.0) == pytest.approx(
        0.2, rel=1e-12, abs=0.0)
    assert value(obs.coron_bandpass, 15000.0) == pytest.approx(
        0.2, rel=1e-12, abs=0.0)
    assert value(obs.coron_bandpass, 1000.0) == pytest.approx(
        0.0, rel=1e-12, abs=0.0)
    # 6000 Angstrom is outside the 6237-6363 Angstrom filter.
    assert value(obs.bandpass, 6000.0) == pytest.approx(
        0.0, rel=1e-12, abs=0.0)


def test_repeated_make_stp_gives_identical_bandpasses(tmp_path):
    # The inputs are written once and used for both make_STP() calls.
    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs = new_observatory()

    def bandpass_values():
        values = []
        for name in ["bandpass", "precoron_bandpass", "coron_bandpass"]:
            curve = getattr(obs, name)
            for wavelength in [6000.0, 6300.0, 15000.0]:
                values.append(float(curve(wavelength * u.AA).value))
        return values

    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    first_run = bandpass_values()
    assert obs.num_mirrors == 4

    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    second_run = bandpass_values()
    assert obs.num_mirrors == 8   # existing accumulation, not changed

    assert second_run == pytest.approx(first_run, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Stage 2: the three bandpasses are built from the registry by
# build_bandpasses(), starting from starting_bandpass.
# ---------------------------------------------------------------------------

def test_build_bandpasses_uses_registry_flags(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_mirror(coating_file=mirror)                 # bandpass, precoron
    obs.add_generic_optic(0.8, coronOnly=True)          # all three
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0),
                       coronOnly=True)                  # bandpass, coron
    bandpass, precoron, coron = throughputs(obs, INSIDE)
    assert bandpass == pytest.approx(0.9 * 0.8 * 0.2, rel=1e-12, abs=0.0)
    assert precoron == pytest.approx(0.9 * 0.8, rel=1e-12, abs=0.0)
    assert coron == pytest.approx(0.8 * 0.2, rel=1e-12, abs=0.0)


def test_build_bandpasses_is_repeatable(tmp_path):
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_mirror(coating_file=mirror)
    obs.add_generic_filter(6300 * u.AA, fbw=0.02)
    obs.add_generic_optic(0.8, coronOnly=True)
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True)
    wavelengths = [3000.0, 6000.0, 6300.0, 13000.0]
    before = [throughputs(obs, w) for w in wavelengths]

    obs.build_bandpasses()
    after = [throughputs(obs, w) for w in wavelengths]

    for old, new in zip(before, after):
        assert new == pytest.approx(old, rel=1e-12, abs=0.0)


def test_starting_bandpass_init_vs_make_stp(tmp_path):
    # A new Observatory starts from a 4000-14000 Angstrom box (width 10000);
    # make_STP() resets it to a 1500-16500 Angstrom box (width 15000).
    obs = new_observatory()
    assert float(obs.starting_bandpass(3000.0 * u.AA).value) == 0.0

    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    assert float(obs.starting_bandpass(3000.0 * u.AA).value) == 1.0


def test_direct_add_after_make_stp_extends_bandpasses(tmp_path):
    telescope_config, instrument_config = (
        write_synthetic_make_stp_inputs(tmp_path))
    obs = new_observatory()
    obs.make_STP(telconfig=telescope_config, escconfig=instrument_config,
                 telpath=tmp_path, escpath=tmp_path)
    bandpass, precoron, coron = throughputs(obs, 6300.0)

    obs.add_generic_optic(0.5)
    new_bandpass, new_precoron, new_coron = throughputs(obs, 6300.0)

    assert new_bandpass == pytest.approx(bandpass * 0.5, rel=1e-12, abs=0.0)
    assert new_precoron == pytest.approx(precoron * 0.5, rel=1e-12, abs=0.0)
    assert new_coron == pytest.approx(coron, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# Stage 3: get_throughput_budget() reports each registered component's
# throughput and the running product, read-only, for one of the three
# bandpasses.
# ---------------------------------------------------------------------------

# Minimum columns; the table may gain extra columns later.
BUDGET_COLUMNS = ["name", "category", "group", "count", "applied",
                  "throughput", "cumulative_throughput"]


def budget_observatory(tmp_path):
    """Five flat synthetic components with different routing.

    m1 0.9 (bandpass, precoron), filter 0.98 at 6237-6363 A (bandpass,
    precoron), optic 0.8 with coronOnly (all three), fpm 0.2 with
    coronOnly (bandpass, coron) and detector_qe 0.7 (bandpass only).
    """
    obs = new_observatory()
    obs.primary_filter = 6300 * u.AA
    mirror = write_flat_curve(tmp_path / "mirror.csv", 0.9)
    table = write_fpm_table(tmp_path / "fpm.fits")
    obs.add_mirror(coating_file=mirror, name="m1", group="telescope")
    obs.add_generic_filter(6300 * u.AA, fbw=0.02,
                           name="filter", group="instrument")
    obs.add_generic_optic(0.8, coronOnly=True,
                          name="optic", group="instrument")
    obs.add_lamD_optic(table, separation_arcsec(obs, 2.0), coronOnly=True,
                       name="fpm", group="instrument")
    add_synthetic_sensor(obs, tmp_path, qe_value=0.7,
                         name="detector_qe", group="detector")
    return obs


def test_throughput_budget_rows_follow_registry(tmp_path):
    obs = budget_observatory(tmp_path)
    budget = obs.get_throughput_budget(6300 * u.AA)
    for column in BUDGET_COLUMNS:
        assert column in budget.columns
    assert list(budget["name"]) == ["m1", "filter", "optic", "fpm",
                                    "detector_qe"]
    assert list(budget["category"]) == ["mirror", "generic_filter",
                                        "generic_optic", "lamD_optic",
                                        "sensor_qe"]
    assert list(budget["group"]) == ["telescope", "instrument", "instrument",
                                     "instrument", "detector"]
    assert list(budget["count"]) == [1, 1, 1, 1, 1]


def test_throughput_budget_full_bandpass(tmp_path):
    # Default bandpass="bandpass": every component is applied.
    obs = budget_observatory(tmp_path)
    budget = obs.get_throughput_budget(6300 * u.AA)
    assert list(budget["applied"]) == [True, True, True, True, True]
    assert list(budget["throughput"]) == pytest.approx(
        [0.9, 0.98, 0.8, 0.2, 0.7], rel=1e-12, abs=0.0)
    assert list(budget["cumulative_throughput"]) == pytest.approx(
        [0.9,
         0.9 * 0.98,
         0.9 * 0.98 * 0.8,
         0.9 * 0.98 * 0.8 * 0.2,
         0.9 * 0.98 * 0.8 * 0.2 * 0.7], rel=1e-12, abs=0.0)


def test_throughput_budget_precoron_keeps_excluded_rows(tmp_path):
    # The FPM and detector QE are not part of precoron_bandpass, but they
    # stay in the table with applied=False and do not change the product.
    obs = budget_observatory(tmp_path)
    budget = obs.get_throughput_budget(6300 * u.AA,
                                       bandpass="precoron_bandpass")
    assert len(budget) == 5
    assert list(budget["applied"]) == [True, True, True, False, False]
    assert list(budget["throughput"]) == pytest.approx(
        [0.9, 0.98, 0.8, 0.2, 0.7], rel=1e-12, abs=0.0)
    optics = 0.9 * 0.98 * 0.8
    assert list(budget["cumulative_throughput"]) == pytest.approx(
        [0.9, 0.9 * 0.98, optics, optics, optics], rel=1e-12, abs=0.0)


def test_throughput_budget_coron_bandpass(tmp_path):
    obs = budget_observatory(tmp_path)
    budget = obs.get_throughput_budget(6300 * u.AA,
                                       bandpass="coron_bandpass")
    assert list(budget["applied"]) == [False, False, True, True, False]
    assert list(budget["cumulative_throughput"]) == pytest.approx(
        [1.0, 1.0, 0.8, 0.8 * 0.2, 0.8 * 0.2], rel=1e-12, abs=0.0)


def test_throughput_budget_final_value_matches_actual_bandpass(tmp_path):
    obs = budget_observatory(tmp_path)
    names = ["bandpass", "precoron_bandpass", "coron_bandpass"]
    for wavelength in [3000.0, 6000.0, 6300.0, 13000.0]:
        actual = throughputs(obs, wavelength)
        for name, actual_value in zip(names, actual):
            budget = obs.get_throughput_budget(wavelength * u.AA,
                                               bandpass=name)
            final = budget["cumulative_throughput"].iloc[-1]
            assert final == pytest.approx(actual_value, rel=1e-12, abs=0.0)


def test_throughput_budget_starts_from_starting_bandpass(tmp_path):
    # 3000 Angstrom is outside the 4000-14000 Angstrom starting box, so the
    # running product is 0 from the first row, even though the mirror's own
    # throughput there is 0.9.
    obs = budget_observatory(tmp_path)
    budget = obs.get_throughput_budget(3000 * u.AA)
    assert budget["throughput"].iloc[0] == pytest.approx(0.9)
    assert list(budget["cumulative_throughput"]) == [0.0] * 5


def test_throughput_budget_wavelength_units(tmp_path):
    # A plain number is read as Angstrom; Quantities may use any length unit.
    obs = budget_observatory(tmp_path)
    budgets = [obs.get_throughput_budget(6300),
               obs.get_throughput_budget(6300 * u.AA),
               obs.get_throughput_budget(630 * u.nm)]
    reference = budgets[0]
    for budget in budgets[1:]:
        assert list(budget["applied"]) == list(reference["applied"])
        for column in ["throughput", "cumulative_throughput"]:
            assert list(budget[column]) == pytest.approx(
                list(reference[column]), rel=1e-12, abs=0.0)


def test_throughput_budget_is_read_only(tmp_path, monkeypatch):
    obs = budget_observatory(tmp_path)
    curves_before = (obs.bandpass, obs.precoron_bandpass,
                     obs.coron_bandpass, obs.starting_bandpass)
    registry_before = obs.throughput_components
    entries_before = [dict(component) for component in registry_before]

    def must_not_rebuild():
        raise AssertionError(
            "get_throughput_budget() rebuilt the bandpasses"
        )

    monkeypatch.setattr(obs, "build_bandpasses", must_not_rebuild)
    for name in ["bandpass", "precoron_bandpass", "coron_bandpass"]:
        obs.get_throughput_budget(6300 * u.AA, bandpass=name)

    # The same curve objects, registry list, entries and entry values.
    curves_after = (obs.bandpass, obs.precoron_bandpass,
                    obs.coron_bandpass, obs.starting_bandpass)
    for after, before in zip(curves_after, curves_before):
        assert after is before
    assert obs.throughput_components is registry_before
    assert len(obs.throughput_components) == len(entries_before)
    for component, before in zip(obs.throughput_components, entries_before):
        assert component.keys() == before.keys()
        for key in before:
            assert component[key] is before[key]


def test_throughput_budget_invalid_bandpass_name(tmp_path):
    obs = budget_observatory(tmp_path)
    with pytest.raises(ValueError, match="bandpass"):
        obs.get_throughput_budget(6300 * u.AA, bandpass="total")


def test_throughput_budget_empty_registry():
    obs = new_observatory()
    budget = obs.get_throughput_budget(6300 * u.AA)
    assert len(budget) == 0
    for column in BUDGET_COLUMNS:
        assert column in budget.columns


# ---------------------------------------------------------------------------
# Stage 4: plot_throughput_budget() draws the registered components, their
# running product and the actual bandpass on one matplotlib axes, without
# changing anything. These tests check the plotted data, not the image.
# ---------------------------------------------------------------------------

def new_axes():
    """Axes on a plain matplotlib Figure: no pyplot and no window."""
    from matplotlib.figure import Figure
    return Figure().add_subplot()


def ydata(line):
    return np.asarray(line.get_ydata(), dtype=float)


def test_plot_budget_creates_figure_without_show(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def must_not_show(*args, **kwargs):
        raise AssertionError(
            "plot_throughput_budget() called plt.show()"
        )

    monkeypatch.setattr(etsc.plt, "show", must_not_show)
    obs = budget_observatory(tmp_path)
    fig, ax, lines = obs.plot_throughput_budget(wavelengths=[6300.0])
    try:
        assert isinstance(fig, Figure)
        assert ax in fig.axes
    finally:
        plt.close(fig)


def test_plot_budget_reuses_supplied_ax(tmp_path):
    obs = budget_observatory(tmp_path)
    ax = new_axes()
    fig, returned_ax, lines = obs.plot_throughput_budget(
        wavelengths=[6300.0], ax=ax)
    assert returned_ax is ax
    assert fig is ax.figure


def test_plot_budget_component_lines_follow_registry(tmp_path):
    obs = budget_observatory(tmp_path)
    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=[6000.0, 6300.0], ax=new_axes())
    components = lines["components"]
    assert [line.get_label() for line in components] == [
        "m1", "filter", "optic", "fpm", "detector_qe"]
    # Each component's own throughput at 6000 and 6300 Angstrom.
    expected = [[0.9, 0.9], [0.0, 0.98], [0.8, 0.8],
                [0.2, 0.2], [0.7, 0.7]]
    for line, values in zip(components, expected):
        assert list(ydata(line)) == pytest.approx(
            values, rel=1e-12, abs=0.0
        )


def test_plot_budget_excluded_components_are_marked(tmp_path):
    obs = budget_observatory(tmp_path)
    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=[6300.0], bandpass="precoron_bandpass",
        ax=new_axes())
    components = lines["components"]
    assert [line.get_label() for line in components] == [
        "m1", "filter", "optic",
        "fpm (not in precoron_bandpass)",
        "detector_qe (not in precoron_bandpass)"]
    # Excluded curves use a different line style from applied ones ...
    applied_styles = {line.get_linestyle() for line in components[:3]}
    excluded_styles = {line.get_linestyle() for line in components[3:]}
    assert len(excluded_styles) == 1
    assert excluded_styles.isdisjoint(applied_styles)
    # ... but keep their own data.
    assert list(ydata(components[3])) == pytest.approx(
        [0.2], rel=1e-12, abs=0.0
    )
    assert list(ydata(components[4])) == pytest.approx(
        [0.7], rel=1e-12, abs=0.0
    )

    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=[6300.0], bandpass="precoron_bandpass",
        show_excluded=False, ax=new_axes())
    assert [line.get_label() for line in lines["components"]] == [
        "m1", "filter", "optic"]


def test_plot_budget_cumulative_values(tmp_path):
    obs = budget_observatory(tmp_path)
    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=[6000.0, 6300.0], ax=new_axes())
    cumulative = lines["cumulative"]
    assert [line.get_label() for line in cumulative] == [
        "cumulative to m1", "cumulative to filter",
        "cumulative to optic", "cumulative to fpm",
        "cumulative to detector_qe"]
    # Running product at 6000 A (outside the filter) and at 6300 A.
    at_6000 = [0.9, 0.0, 0.0, 0.0, 0.0]
    at_6300 = [0.9,
               0.9 * 0.98,
               0.9 * 0.98 * 0.8,
               0.9 * 0.98 * 0.8 * 0.2,
               0.9 * 0.98 * 0.8 * 0.2 * 0.7]
    for line, low, high in zip(cumulative, at_6000, at_6300):
        assert list(ydata(line)) == pytest.approx(
            [low, high], rel=1e-12, abs=0.0
        )
    # Each cumulative curve shares its component's colour.
    for component, total in zip(lines["components"], cumulative):
        assert total.get_color() == component.get_color()

    # coron_bandpass: only optic and fpm are applied.
    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=[6300.0], bandpass="coron_bandpass", ax=new_axes())
    assert [line.get_label() for line in lines["cumulative"]] == [
        "cumulative to optic", "cumulative to fpm"]
    assert [ydata(line)[0] for line in lines["cumulative"]] == (
        pytest.approx([0.8, 0.8 * 0.2], rel=1e-12, abs=0.0)
    )


def test_plot_budget_final_matches_actual_bandpass(tmp_path):
    obs = budget_observatory(tmp_path)
    wavelengths = np.arange(3000.0, 14001.0, 50.0)
    for name in ["bandpass", "precoron_bandpass", "coron_bandpass"]:
        fig, ax, lines = obs.plot_throughput_budget(
            wavelengths=wavelengths, bandpass=name, ax=new_axes())
        actual = getattr(obs, name)(wavelengths * u.AA).value
        assert list(ydata(lines["cumulative"][-1])) == pytest.approx(
            list(actual), rel=1e-12, abs=0.0
        )
        assert list(ydata(lines["final"])) == pytest.approx(
            list(actual), rel=1e-12, abs=0.0
        )
        assert list(lines["final"].get_xdata()) == pytest.approx(
            list(wavelengths), rel=1e-12, abs=0.0
        )


def test_plot_budget_switches(tmp_path):
    obs = budget_observatory(tmp_path)
    w = [6300.0]
    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=w, ax=new_axes())
    # 5 component curves + 5 cumulative curves + the final bandpass.
    assert len(ax.lines) == 11

    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=w, show_components=False, ax=new_axes())
    assert lines["components"] == []
    assert len(lines["cumulative"]) == 5
    assert lines["final"] is not None

    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=w, show_cumulative=False, ax=new_axes())
    assert len(lines["components"]) == 5
    assert lines["cumulative"] == []

    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=w, show_final=False, ax=new_axes())
    assert lines["final"] is None

    fig, ax, lines = obs.plot_throughput_budget(
        wavelengths=w, show_components=False, show_cumulative=False,
        show_final=False, ax=new_axes())
    assert lines == {"components": [], "cumulative": [], "final": None}
    assert len(ax.lines) == 0


def test_plot_budget_default_wavelength_grid(tmp_path):
    from synphot import SpectralElement
    from synphot.models import Box1D

    # A bare Observatory starts from the 4000-14000 Angstrom box.
    obs = budget_observatory(tmp_path)
    fig, ax, lines = obs.plot_throughput_budget(
        show_components=False, show_cumulative=False, ax=new_axes())
    x = np.asarray(lines["final"].get_xdata(), dtype=float)
    assert len(x) == 10001
    assert x[0] == 4000.0
    assert x[-1] == 14000.0
    assert np.all(np.diff(x) == 1.0)

    # The grid follows the starting box, e.g. make_STP()'s 1500-16500 A.
    obs.starting_bandpass = SpectralElement(
        Box1D, amplitude=1, x_0=9000, width=15000)
    fig, ax, lines = obs.plot_throughput_budget(
        show_components=False, show_cumulative=False, ax=new_axes())
    x = np.asarray(lines["final"].get_xdata(), dtype=float)
    assert len(x) == 15001
    assert x[0] == 1500.0
    assert x[-1] == 16500.0


def test_plot_budget_wavelength_units(tmp_path):
    # A plain list is read as Angstrom; Quantities may use any length unit.
    obs = budget_observatory(tmp_path)
    angstrom = [6240.0, 6300.0, 6360.0]
    inputs = [angstrom,
              np.array(angstrom) * u.AA,
              np.array(angstrom) / 10.0 * u.nm]
    results = [obs.plot_throughput_budget(wavelengths=w, ax=new_axes())[2]
               for w in inputs]
    reference = results[0]
    for lines in results[1:]:
        for kind in ["components", "cumulative"]:
            for line, ref in zip(lines[kind], reference[kind]):
                assert list(line.get_xdata()) == pytest.approx(
                    list(ref.get_xdata()), rel=1e-12, abs=0.0
                )
                assert list(ydata(line)) == pytest.approx(
                    list(ydata(ref)), rel=1e-12, abs=0.0
                )

    for not_1d in [6300.0, [[6300.0, 6310.0]]]:
        with pytest.raises(ValueError, match="1-D"):
            obs.plot_throughput_budget(wavelengths=not_1d, ax=new_axes())


def test_plot_budget_invalid_bandpass_name(tmp_path):
    obs = budget_observatory(tmp_path)
    with pytest.raises(ValueError, match="bandpass"):
        obs.plot_throughput_budget(
            wavelengths=[6300.0], bandpass="total", ax=new_axes())


def test_plot_budget_is_read_only(tmp_path, monkeypatch):
    obs = budget_observatory(tmp_path)
    curves_before = (obs.bandpass, obs.precoron_bandpass,
                     obs.coron_bandpass, obs.starting_bandpass)
    registry_before = obs.throughput_components
    entries_before = [dict(component) for component in registry_before]

    def must_not_rebuild():
        raise AssertionError(
            "plot_throughput_budget() rebuilt the bandpasses"
        )

    monkeypatch.setattr(obs, "build_bandpasses", must_not_rebuild)
    for name in ["bandpass", "precoron_bandpass", "coron_bandpass"]:
        obs.plot_throughput_budget(
            wavelengths=[6000.0, 6300.0], bandpass=name, ax=new_axes())

    # The same curve objects, registry list, entries and entry values.
    curves_after = (obs.bandpass, obs.precoron_bandpass,
                    obs.coron_bandpass, obs.starting_bandpass)
    for after, before in zip(curves_after, curves_before):
        assert after is before
    assert obs.throughput_components is registry_before
    assert len(obs.throughput_components) == len(entries_before)
    for component, before in zip(obs.throughput_components,
                                 entries_before):
        assert component.keys() == before.keys()
        for key in before:
            assert component[key] is before[key]


# ---------------------------------------------------------------------------
# Stage 5: count rates from the real make_observation() path, checked
# against scaling laws (magnitude, detector QE, collecting area and a flat
# throughput factor). Only ratios are tested; absolute count rates depend
# on reference spectra and assumptions that these tests do not fix.
# ---------------------------------------------------------------------------

def use_offline_reference_spectra(monkeypatch):
    """Stand-ins for the Vega spectrum and Johnson V filter.

    make_observation() always loads both. Replacing them with flat
    synthetic versions avoids a network download and makes the tests
    independent of which Vega file a synphot version uses. Ratios of
    source count rates do not depend on them.
    """
    from synphot.models import ConstFlux1D, Empirical1D

    flat_vega = etsc.SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag)
    flat_v_band = etsc.SpectralElement(
        Empirical1D, points=[4500.0, 4501.0, 6499.0, 6500.0],
        lookup_table=[0.0, 1.0, 1.0, 0.0])

    def fake_from_vega(**kwargs):
        return flat_vega

    def fake_from_filter(filter_name, **kwargs):
        return flat_v_band

    monkeypatch.setattr(etsc.SourceSpectrum, "from_vega", fake_from_vega)
    monkeypatch.setattr(etsc.SpectralElement, "from_filter",
                        fake_from_filter)


def countrate_observatory(directory, diameter=3.0 * u.m, qe=None, k=None):
    """A small synthetic system ready for make_observation().

    Mirror 0.9 and the 6300 A, 2% filter; optionally a flat throughput
    factor k and a flat detector QE (both CSV curves).
    """
    from synphot.models import Box1D, ConstFlux1D

    obs = etsc.Observatory("countrate", diameter, 36 * u.m)
    # Test-only speed-up: a 6100-6500 A starting box instead of the usual
    # 4000-14000 A one. The 6237-6363 A filter makes every bandpass zero
    # outside that range anyway, so the count rates are the same, but
    # synphot samples about 50,000 instead of about 1,000,000 points.
    obs.starting_bandpass = etsc.SpectralElement(
        Box1D, amplitude=1, x_0=6300, width=400)
    obs.build_bandpasses()

    obs.primary_filter = 6300 * u.AA
    obs.add_mirror(coating_file=write_flat_curve(directory / "mirror.csv",
                                                 0.9))
    obs.add_generic_filter(6300 * u.AA, fbw=0.02)
    if k is not None:
        obs.add_transmissive_optic(write_flat_curve(directory / "k.csv", k),
                                   wave_unit="nm")
    if qe is not None:
        add_synthetic_sensor(obs, directory, qe_value=qe)

    # The minimum make_observation() reads besides the bandpass.
    obs.background_spectrum = etsc.SourceSpectrum(ConstFlux1D,
                                                  amplitude=22 * u.ABmag)
    obs.resel = 1.0 * u.arcsec**2
    obs.rawDH_contrast = 1e-8
    obs.set_generic_source(1.0, 0.0)
    return obs


def source_rate(obs, hoststarflux, planetdeltamag=0.0, flux_units="AB"):
    """Source count rate from the real make_observation() path."""
    obs.make_observation(hoststarflux=hoststarflux,
                         planetdeltamag=planetdeltamag,
                         flux_units=flux_units, bg_flux=22.5, exobg_flux=22)
    return obs.source_counts.value


def test_countrate_magnitude_scaling_ab(tmp_path, monkeypatch):
    use_offline_reference_spectra(monkeypatch)
    obs = countrate_observatory(tmp_path)
    rate_0 = source_rate(obs, 0.0)
    rate_3 = source_rate(obs, 3.0)
    rate_5 = source_rate(obs, 5.0)
    assert rate_3 / rate_0 == pytest.approx(10**(-1.2), rel=1e-10)
    assert rate_5 / rate_0 == pytest.approx(0.01, rel=1e-10)
    # Only the total magnitude matters, not how it is split.
    assert source_rate(obs, 0.0, planetdeltamag=3.0) == pytest.approx(
        rate_3, rel=1e-10)


def test_countrate_magnitude_scaling_vega(tmp_path, monkeypatch):
    # Uses the flat stand-in Vega spectrum; no absolute Vega rate is
    # checked, only the magnitude scaling.
    use_offline_reference_spectra(monkeypatch)
    obs = countrate_observatory(tmp_path)
    rate_0 = source_rate(obs, 0.0, flux_units="vega")
    rate_3 = source_rate(obs, 3.0, flux_units="vega")
    rate_5 = source_rate(obs, 5.0, flux_units="vega")
    assert rate_3 / rate_0 == pytest.approx(10**(-1.2), rel=1e-10)
    assert rate_5 / rate_0 == pytest.approx(0.01, rel=1e-10)
    assert source_rate(obs, 0.0, planetdeltamag=3.0,
                       flux_units="vega") == pytest.approx(rate_3,
                                                           rel=1e-10)


def test_countrate_includes_detector_qe(tmp_path, monkeypatch):
    # The normal ETC count rate uses self.bandpass, which includes the
    # detector QE, so it is in photoelectrons per second. The same system
    # without a detector gives photons per second. make_observation()
    # itself is unchanged.
    use_offline_reference_spectra(monkeypatch)
    without_qe = countrate_observatory(tmp_path)
    with_qe = countrate_observatory(tmp_path, qe=0.7)
    ratio = source_rate(with_qe, 0.0) / source_rate(without_qe, 0.0)
    assert ratio == pytest.approx(0.7, rel=1e-10)

    # With no FPM here, the only difference is the detector QE, which
    # precoron_bandpass leaves out.
    for wavelength in [6240.0, 6300.0, 6360.0]:
        w = wavelength * u.AA
        assert with_qe.precoron_bandpass(w).value == pytest.approx(
            without_qe.bandpass(w).value, rel=1e-12, abs=0.0)


def test_countrate_scales_with_collecting_area(tmp_path, monkeypatch):
    # Collecting area is pi * (D/2)**2, so sqrt(2) times the diameter
    # doubles it.
    use_offline_reference_spectra(monkeypatch)
    small = countrate_observatory(tmp_path, diameter=3.0 * u.m)
    large = countrate_observatory(tmp_path, diameter=3.0 * np.sqrt(2) * u.m)
    ratio = source_rate(large, 0.0) / source_rate(small, 0.0)
    assert ratio == pytest.approx(2.0, rel=1e-10)


def test_countrate_scales_with_flat_throughput(tmp_path, monkeypatch):
    use_offline_reference_spectra(monkeypatch)
    base = countrate_observatory(tmp_path)
    with_k = countrate_observatory(tmp_path, k=0.5)
    ratio = source_rate(with_k, 0.0) / source_rate(base, 0.0)
    assert ratio == pytest.approx(0.5, rel=1e-10)
