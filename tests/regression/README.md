# Historical UM regression check

This folder holds an **opt-in** regression check that compares the current ETC
against a recorded baseline of the **unmodified** ETC, for the Ultramarine (UM)
configuration. It exists to prove that refactoring how the throughput curves
are assembled (the component-throughput-budget work) does not change the
science results.

It is **not** part of the normal `pytest` run or CI. None of the files here
start with `test_`, so pytest never collects them.

## Why it is separate from the normal tests

At the baseline commit (`69da4fe`, `origin/develop`), no configuration runs
end-to-end in the repository's current dependency environment:

- **Default STP** fails with `KeyError: 'dichroic'`: `make_STP()` reads
  `[arm_a.optics.dichroic]`, which `config_stp_esc` does not define.
- **UM** fails with synphot 1.4 and newer: the `config_um` coating file
  `coatings/qe_ag_fss99_profile_mirror.fits` has no `TUNIT` unit headers, and
  synphot 1.4+ ignores the `wave_unit='nm'` argument the ETC passes for it.

Both are pre-existing issues and are not fixed here. A committed test that
needs UM would therefore fail in CI for reasons unrelated to the code under
test, so this check is run by hand instead.

## The historical environment

This check must be run in a pinned environment that can still run the
unmodified ETC. **These versions reproduce historical behaviour. They are NOT
the project's dependency requirements**, and nothing here changes
`pyproject.toml`, `requirements.txt` or CI.

`check_um_regression.py` compares every version below **exactly** with the
versions recorded in the baseline. Any difference gives exit status 2 (not a
valid historical comparison), even if every scientific value matches.

| Package | Exact version |
|---|---|
| Python | 3.12.15 |
| numpy | 1.26.4 |
| astropy | 6.1.7 |
| synphot | 1.3.post0 |
| scipy | 1.17.1 |
| pandas | 3.0.6 |
| utils_config | `1273b742512cf78892e0cb59a57b816810e6b681` |
| config_stp | `b13e31bf4372ec5430a328319ae7b07cc6030aaa` |
| config_stp_esc | `4c048724596aa3be34c8059bdcfd2d67ff5b3ed0` |
| config_um | `838ad1ca0134a9c5d371399de0fe5b7779d84feb` |
| config_um_esc | `6b978bd986fa684c17a8fcb57ea16be54dc5ba06` |

The same values are recorded in the `#` header of both baseline files.

To create it (from the repository root; the environment name is arbitrary):

```bash
conda create -y -n etc-historical python=3.12.15 pip
conda activate etc-historical

C=/tmp/stp-etc-historical-constraints.txt
printf '%s\n' \
  numpy==1.26.4 \
  astropy==6.1.7 \
  synphot==1.3.post0 \
  scipy==1.17.1 \
  pandas==3.0.6 \
  > $C

pip install -c $C \
  numpy==1.26.4 \
  astropy==6.1.7 \
  synphot==1.3.post0 \
  scipy==1.17.1 \
  pandas==3.0.6
```

```bash
pip install -c $C \
  "utils_config @ git+https://github.com/uasal/utils_config.git@1273b742512cf78892e0cb59a57b816810e6b681" \
  "config_stp @ git+https://github.com/uasal/config_stp.git@b13e31bf4372ec5430a328319ae7b07cc6030aaa" \
  "config_stp_esc @ git+https://github.com/uasal/config_stp_esc.git@4c048724596aa3be34c8059bdcfd2d67ff5b3ed0" \
  "config_um @ git+https://github.com/uasal/config_um.git@838ad1ca0134a9c5d371399de0fe5b7779d84feb" \
  "config_um_esc @ git+https://github.com/uasal/config_um_esc.git@6b978bd986fa684c17a8fcb57ea16be54dc5ba06"

pip install -c $C -e .
```

Install the ETC with `-e .` (editable) so the check runs the code in your
working tree, not an older installed copy.

## Files

| File | Purpose |
|---|---|
| `um_reference.py` | Shared setup: the UM configuration, the observation inputs, and the list of summary quantities. Used by both scripts so they run the ETC identically. |
| `make_um_baseline.py` | Writes the baseline. Run **once**, on unmodified code. |
| `check_um_regression.py` | Compares the current ETC with the baseline. |
| `../test_data/bandpass_baseline_um.csv` | `bandpass`, `precoron_bandpass` and `coron_bandpass` at fixed wavelengths. |
| `../test_data/um_baseline_summary.csv` | Peaks, widths, `num_mirrors`, `qe_curves`, count rates, SNR, integration time, saturation time. |

## Creating the baseline

```bash
python tests/regression/make_um_baseline.py
```

This is done **once**, in the historical environment, on clean, unmodified
ETC code. Once generated, these fixed baseline files are not rerun or
replaced as part of the refactor: the whole point is to compare new code
against them.

The script refuses to overwrite existing baseline files and refuses to run if
`src/` has uncommitted changes. If the baseline ever needs to be re-recorded
(for example, after a deliberate, reviewed change to the science), agree that
first, delete both files in a separate commit that explains why, and
regenerate them on clean code.

## Running the check

```bash
python tests/regression/check_um_regression.py
```

It takes about 7–8 minutes: the UM bandpasses have ~5.5 million internal
sample points, which makes synphot's integrals and `make_observation()` slow.
Cheap checks on the baseline files run first.

| Exit status | Meaning |
|---|---|
| 0 | Valid historical check: every scientific value matches, in the baseline environment. |
| 1 | Regression: a scientific value or a recorded ETC input differs, or the baseline files are malformed or do not come from the same run. |
| 2 | Scientific values match, but the environment differs from the baseline's, so this is not a valid historical comparison. |

**What is compared:** the three curves at the stored wavelengths, and every
summary row. Numbers use a relative tolerance of `1e-12` with no absolute
allowance, so values that were exactly zero must stay exactly zero. Text rows
(the `qe_curves` entries) must match exactly, and units must match.

**What is only reported:** when the baseline was made, which ETC commit made
it, and whether the current `src/` has uncommitted changes. The ETC commit is
expected to differ, because the point is to test newer commits.

## What each part of the baseline checks

The curve file samples physical wavelengths (in Ångström) chosen when the
baseline was made:

- 1000–17000 Å every 100 Å (coverage, including outside the curves);
- 6200–6400 Å every 0.5 Å (the 6300 Å, 2% science filter);
- −0.1, −0.01, 0, +0.01 and +0.1 Å around the edges at 1500 and 16500 Å
  (the `make_STP()` starting placeholder) and 6237 and 6363 Å (the filter);
- the peak wavelength of each of the three curves.

The checker reads these wavelengths from the file. It never rebuilds the grid
from the current code, so a change in the code cannot silently change what is
being checked. The full curves are not stored: at ~5.5 million points each
they would be hundreds of megabytes.

Together the files check:

- **Sampled curves:** `bandpass`, `precoron_bandpass` and `coron_bandpass`
  directly, at the stored wavelengths.
- **`wpeak`, `tpeak` and `rectwidth`:** additional scalar checks on all three
  curves, computed by synphot over each curve's full sampling.
- **Count rates, SNR, integration time and saturation time:** an integrated,
  end-to-end check of the final `self.bandpass`, which is the curve
  `make_observation()` uses. They do not involve `precoron_bandpass` or
  `coron_bandpass`.
