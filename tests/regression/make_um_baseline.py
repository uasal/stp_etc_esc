"""Write the historical UM regression baseline (run ONCE, on unmodified ETC code).

Usage, from the repository root, in the pinned historical environment
(see README.md in this folder):

    python tests/regression/make_um_baseline.py

Writes:
    tests/test_data/bandpass_baseline_um.csv   compact curve samples
    tests/test_data/um_baseline_summary.csv    peaks, widths, count rates, SNR, ...

It refuses to overwrite an existing baseline and refuses to run if the ETC
source (src/) has uncommitted changes. This is the ONLY place the wavelength
grid is constructed; check_um_regression.py reads the stored wavelengths.
"""

import csv
import sys

import um_reference as ref

# Physical features of the UM bandpasses that the grid must resolve:
# the make_STP() starting placeholder spans 1500-16500 Angstrom and the
# science filter spans 6237-6363 Angstrom (6300 Angstrom, 2% bandwidth).
EDGES_ANGSTROM = [1500.0, 6237.0, 6363.0, 16500.0]
EDGE_OFFSETS_ANGSTROM = [-0.1, -0.01, 0.0, 0.01, 0.1]

GRID_DESCRIPTION = ("grid: 1000-17000 A every 100 A; 6200-6400 A every 0.5 A; "
                    "edges 1500, 6237, 6363, 16500 A at -0.1, -0.01, 0, +0.01, +0.1 A; "
                    "the three wpeak wavelengths. Sorted, duplicates removed.")


def hybrid_wavelength_grid(peak_wavelengths):
    """The approved fixed grid, in Angstrom. Built only here, at baseline time."""
    wavelengths = []
    wavelengths += [1000.0 + 100.0 * i for i in range(161)]   # 1000-17000, every 100
    wavelengths += [6200.0 + 0.5 * i for i in range(401)]     # 6200-6400, every 0.5
    for edge in EDGES_ANGSTROM:
        wavelengths += [edge + offset for offset in EDGE_OFFSETS_ANGSTROM]
    wavelengths += list(peak_wavelengths)
    return sorted(set(wavelengths))


def write_header(handle, metadata_lines, extra_lines=()):
    for line in metadata_lines:
        handle.write(f"# {line}\n")
    for line in extra_lines:
        handle.write(f"# {line}\n")


def main():
    for path in [ref.CURVE_BASELINE, ref.SUMMARY_BASELINE]:
        if path.exists():
            sys.exit(f"REFUSING: {path} already exists. The baseline is never "
                     "regenerated silently; ask before re-recording it.")
    if ref.etc_source_is_dirty():
        sys.exit("REFUSING: src/ has uncommitted changes. The baseline must come "
                 "from unmodified ETC code.")

    # Captured once so both baseline files carry identical provenance,
    # including the same generated_utc (the start time of this run).
    metadata_lines = ref.metadata_lines()

    ref.progress("Building UM observatory with make_STP ...")
    obs = ref.build_um_observatory()

    # Evaluate the curves before compute_summary(), which goes on to set up
    # an observation on the same object.
    ref.progress("Computing summary quantities ...")
    peak_wavelengths = [getattr(obs, name).wpeak().to_value("AA")
                        for name in ref.BANDPASS_NAMES]
    wavelengths = hybrid_wavelength_grid(peak_wavelengths)
    curves = ref.evaluate_bandpasses(obs, wavelengths)
    summary_rows = ref.compute_summary(obs)

    ref.progress(f"Writing {ref.CURVE_BASELINE} ({len(wavelengths)} rows) ...")
    with open(ref.CURVE_BASELINE, "w", newline="") as handle:
        write_header(handle, metadata_lines,
                     [GRID_DESCRIPTION, "values: repr() of Python floats (exact round trip)"])
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["wavelength_angstrom"] + ref.BANDPASS_NAMES)
        for i, wavelength in enumerate(wavelengths):
            writer.writerow([repr(wavelength)] +
                            [repr(float(curves[name][i])) for name in ref.BANDPASS_NAMES])

    ref.progress(f"Writing {ref.SUMMARY_BASELINE} ({len(summary_rows)} rows) ...")
    with open(ref.SUMMARY_BASELINE, "w", newline="") as handle:
        write_header(handle, metadata_lines,
                     ["values: repr() of Python floats (exact round trip); "
                      "unit 'text' rows are compared as exact strings"])
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["quantity", "value", "unit"])
        for name, value, unit in summary_rows:
            writer.writerow([name, value if unit == "text" else repr(value), unit])

    ref.progress("Done.")


if __name__ == "__main__":
    main()
