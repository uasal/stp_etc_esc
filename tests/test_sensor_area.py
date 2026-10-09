"""add_sensor() must keep a caller-provided sensor_area and must not
assume a default when none is given.

Uses small synthetic sensor curves written to tmp_path, so no configuration
package data, FITS files or network access are needed.
"""
import astropy.units as u
import numpy as np
import pytest

from stp_etc_esc import ExposureTimeSNRCalculatorESC as etsc

PIXEL_SIZE = 4.0 * u.um / u.pix


def write_csv(path, header, rows):
    lines = [header] + [f"{x},{y}" for x, y in rows]
    path.write_text("\n".join(lines) + "\n")
    return path.name


def make_observatory(tmp_path):
    """Observatory with a synthetic arm_a sensor configuration."""
    sensor = {
        "pixel_size": "4.0e-6m",
        "gain_curve": write_csv(tmp_path / "gain.csv", "setting,gain",
                                [(0, 2.0), (200, 1.0)]),
        "dark_current": write_csv(tmp_path / "dark.csv", "temp,dark",
                                  [(-30, 0.001), (30, 0.1)]),
        "read_noise": write_csv(tmp_path / "read_noise.csv", "setting,rn",
                                [(0, 3.0), (200, 1.0)]),
        "well_depth": write_csv(tmp_path / "well_depth.csv", "setting,wd",
                                [(0, 50000), (200, 10000)]),
    }
    write_csv(tmp_path / "qe.csv", "wavelength_nm,qe",
              [(300, 0.5), (1100, 0.5)])

    obs = etsc.Observatory("synthetic", 3.0 * u.m, 36.0 * u.m)
    obs.instrument_config = {"common_params": {"arm_a": {"sensor": sensor}}}
    obs.instrument_SP = tmp_path
    return obs


@pytest.fixture
def observatory(tmp_path):
    return make_observatory(tmp_path)


def add_sensor(obs, **kwargs):
    qe_file = (obs.instrument_SP / "qe.csv").as_posix()
    obs.add_sensor(qe_file, gain_setting=100, sensor_temp=0 * u.Celsius,
                   **kwargs)


def test_custom_sensor_area_is_kept(observatory):
    area = 1000 * u.um * 2000 * u.um
    add_sensor(observatory, sensor_area=area)

    assert observatory.sensor_area == area
    expected = (area / PIXEL_SIZE**2).to(u.pix**2).value
    assert observatory.num_pixels.value == pytest.approx(expected,
                                                         rel=1e-12)


def test_custom_sensor_area_in_other_units(observatory):
    add_sensor(observatory, sensor_area=1 * u.mm**2)

    assert observatory.sensor_area == 1 * u.mm**2
    assert observatory.num_pixels.value == pytest.approx(62500, rel=1e-12)


def test_missing_sensor_area_raises(observatory):
    with pytest.raises(ValueError, match="sensor_area must be provided"):
        add_sensor(observatory, sensor_area=None)


def test_sensor_area_does_not_change_other_detector_values(tmp_path):
    def build(subdir, **kwargs):
        path = tmp_path / subdir
        path.mkdir()
        obs = make_observatory(path)
        add_sensor(obs, **kwargs)
        return obs

    small = build("small", sensor_area=1000 * u.um * 2000 * u.um)
    large = build("large", sensor_area=1 * u.mm**2)

    for name in ("gain", "dark_current", "read_noise", "well_depth",
                 "pixel_size", "plate_scale", "sensor_temp"):
        assert getattr(large, name) == getattr(small, name), name
    wavelengths = np.array([3000, 6300, 10000]) * u.AA
    assert np.array_equal(large.bandpass(wavelengths),
                          small.bandpass(wavelengths))
