"""Compare the current ETC against the historical UM baseline.

Usage, from the repository root, in the pinned historical environment
(see README.md in this folder):

    python tests/regression/check_um_regression.py

The regression criteria are the scientific values only: the three bandpass
curves at the stored wavelengths, and the summary quantities (peaks, widths,
num_mirrors, qe_curves, count rates, SNR, integration and saturation time).
Numbers must agree to RTOL with no absolute allowance, so a baseline zero
must stay exactly zero. Text rows must match exactly.

Wavelengths are read from bandpass_baseline_um.csv. They are never
reconstructed from the current ETC or configuration.

Exit status:
    0  valid historical regression check: everything passed, in the same
       environment as the baseline
    1  regression: a scientific value or a recorded input differs, or the
       baseline files are malformed or do not belong together
    2  scientific values agree, but the environment differs from the
       baseline's, so this is not a valid historical comparison

Takes ~7-8 minutes because the UM count rates are slow. All cheap checks on
the baseline files run first, before the slow ETC calculation.
"""

import csv
import sys

import numpy as np

import um_reference as ref

# Same code and environment reproduce the baseline bit for bit; this only
# allows harmless last-digit differences, e.g. from multiplication order.
RTOL = 1e-12

CURVE_COLUMNS = ["wavelength_angstrom"] + ref.BANDPASS_NAMES
SUMMARY_COLUMNS = ["quantity", "value", "unit"]

# Recorded for information only. The ETC commit is EXPECTED to differ: the
# point is to test later (refactored) commits against this baseline.
REPORT_ONLY_KEYS = ["generated_utc", "stp_etc_esc_commit", "stp_etc_esc_src_dirty"]

# The pinned historical environment. A mismatch makes the result not comparable.
ENVIRONMENT_KEYS = ["python", "numpy", "astropy", "synphot", "scipy", "pandas",
                    "config_stp_commit", "config_stp_esc_commit", "config_um_commit",
                    "config_um_esc_commit", "utils_config_commit"]

# The ETC inputs. If these change, the baseline no longer describes this run.
INPUT_KEYS = ["observatory", "make_STP_args", "set_generic_source_args",
              "set_background_args", "make_observation_args",
              "calc_SNR_args", "calc_int_time_args"]


def fail_baseline(message):
    print(f"RESULT: baseline problem: {message} (exit 1)")
    sys.exit(1)


def read_csv(path, expected_columns):
    """Read a baseline CSV (skipping '#' lines); exit 1 if it is malformed."""
    if not path.exists():
        fail_baseline(f"{path} does not exist")
    with open(path) as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames != expected_columns:
        fail_baseline(f"{path.name} columns are {reader.fieldnames}, "
                      f"expected {expected_columns}")
    rows = list(reader)
    if not rows:
        fail_baseline(f"{path.name} has no data rows")
    for number, row in enumerate(rows, start=1):
        if None in row or any(value is None or value == "" for value in row.values()):
            fail_baseline(f"{path.name} data row {number} has missing or extra fields")
    return rows


def compare_numbers(baseline, current):
    """Return (max absolute difference, max fractional difference, passed).

    NaN on both sides counts as agreement (calc_int_time reports an
    infeasible integration time as NaN); NaN on one side only is a failure.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    both_nan = np.isnan(baseline) & np.isnan(current)
    abs_diff = np.where(both_nan, 0.0, np.abs(current - baseline))
    frac_diff = np.zeros_like(abs_diff)
    nonzero = (baseline != 0) & ~both_nan
    frac_diff[nonzero] = abs_diff[nonzero] / np.abs(baseline[nonzero])
    frac_diff[(baseline == 0) & (current != 0)] = np.inf   # a zero that stopped being zero
    within = abs_diff <= RTOL * np.abs(baseline)
    passed = bool(np.all(within | both_nan))
    return float(np.nanmax(abs_diff)), float(np.nanmax(frac_diff)), passed


def check_baseline_files_belong_together():
    """Both files must come from one clean baseline run; exit 1 otherwise."""
    curve_meta = ref.read_metadata(ref.CURVE_BASELINE)
    summary_meta = ref.read_metadata(ref.SUMMARY_BASELINE)
    keys = ["generated_utc", "stp_etc_esc_commit", "stp_etc_esc_src_dirty"]
    keys += ENVIRONMENT_KEYS + INPUT_KEYS
    problems = []
    for key in keys:
        if key not in curve_meta or key not in summary_meta:
            problems.append(f"{key} missing from a baseline header")
        elif curve_meta[key] != summary_meta[key]:
            problems.append(f"{key}: curve file={curve_meta[key]}  "
                            f"summary file={summary_meta[key]}")
    if summary_meta.get("stp_etc_esc_src_dirty") != "False":
        problems.append("stp_etc_esc_src_dirty is not False: baseline was not "
                        "generated from clean ETC source")
    for problem in problems:
        print(f"  {problem}")
    if problems:
        fail_baseline("the two baseline files are not from the same clean run")


def check_provenance():
    """Print provenance; return (environment ok, inputs ok)."""
    baseline = ref.read_metadata(ref.SUMMARY_BASELINE)
    current = ref.provenance()

    print("Provenance (report only):")
    for key in REPORT_ONLY_KEYS:
        print(f"  {key:26s} baseline={baseline.get(key)}  current={current.get(key)}")

    environment_ok = True
    print("\nPinned historical environment:")
    for key in ENVIRONMENT_KEYS:
        same = baseline.get(key) == current.get(key)
        environment_ok = environment_ok and same
        status = "same" if same else "DIFFERENT"
        print(f"  {status:9s} {key:24s} baseline={baseline.get(key)}  current={current.get(key)}")

    inputs_ok = True
    print("\nETC inputs:")
    for key in INPUT_KEYS:
        same = baseline.get(key) == current.get(key)
        inputs_ok = inputs_ok and same
        status = "same" if same else "DIFFERENT"
        print(f"  {status:9s} {key:24s} baseline={baseline.get(key)}  current={current.get(key)}")
    return environment_ok, inputs_ok


def main():
    # Cheap checks on the baseline files, before the slow ETC calculation.
    print("Baseline files:")
    curve_rows = read_csv(ref.CURVE_BASELINE, CURVE_COLUMNS)
    summary_rows = read_csv(ref.SUMMARY_BASELINE, SUMMARY_COLUMNS)
    check_baseline_files_belong_together()
    print(f"  consistent: {len(curve_rows)} curve rows, {len(summary_rows)} summary rows\n")

    environment_ok, inputs_ok = check_provenance()
    results = []   # (passed, max abs diff, max frac diff, label)

    # 1. Curves, evaluated at exactly the stored wavelengths.
    ref.progress("\nBuilding UM observatory with make_STP ...")
    obs = ref.build_um_observatory()

    wavelengths = [float(row["wavelength_angstrom"]) for row in curve_rows]
    current_curves = ref.evaluate_bandpasses(obs, wavelengths)
    for name in ref.BANDPASS_NAMES:
        baseline_values = [float(row[name]) for row in curve_rows]
        max_abs, max_frac, passed = compare_numbers(baseline_values, current_curves[name])
        results.append((passed, max_abs, max_frac,
                        f"curve {name} at {len(wavelengths)} stored wavelengths"))

    # 2. Summary quantities.
    ref.progress("Computing summary quantities ...")
    current_summary = {name: (value, unit) for name, value, unit in ref.compute_summary(obs)}
    baseline_names = [row["quantity"] for row in summary_rows]

    for row in summary_rows:
        name, baseline_value, unit = row["quantity"], row["value"], row["unit"]
        if name not in current_summary:
            results.append((False, np.nan, np.nan, f"{name}: missing from current run"))
            continue
        current_value, current_unit = current_summary[name]
        if unit == "text":
            passed = current_value == baseline_value
            results.append((passed, np.nan, np.nan,
                            f"{name}: {baseline_value!r} -> {current_value!r}"))
        else:
            max_abs, max_frac, passed = compare_numbers([float(baseline_value)],
                                                        [current_value])
            if current_unit != unit:
                passed = False
            results.append((passed, max_abs, max_frac,
                            f"{name}: {baseline_value} -> {current_value!r} "
                            f"[{unit} -> {current_unit}]"))

    for name in current_summary:
        if name not in baseline_names:
            results.append((False, np.nan, np.nan, f"{name}: not in baseline"))

    # 3. Report.
    print(f"\n{'result':6s} {'max abs diff':>13s} {'max frac diff':>13s}  quantity")
    for passed, max_abs, max_frac, label in results:
        print(f"{'PASS' if passed else 'FAIL':6s} {max_abs:13.3e} {max_frac:13.3e}  {label}")

    n_failed = sum(1 for result in results if not result[0])
    print(f"\nScientific values: {len(results) - n_failed} passed, {n_failed} failed "
          f"(rtol={RTOL}, atol=0)")
    print(f"ETC inputs match baseline: {inputs_ok}")
    print(f"Pinned environment matches baseline: {environment_ok}")

    if n_failed or not inputs_ok:
        print("RESULT: REGRESSION (exit 1)")
        sys.exit(1)
    if not environment_ok:
        print("RESULT: values agree, but the environment differs from the baseline, "
              "so this is not a valid historical comparison (exit 2)")
        sys.exit(2)
    print("RESULT: PASS (exit 0)")


if __name__ == "__main__":
    main()
