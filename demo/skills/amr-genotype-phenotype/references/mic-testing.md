# MIC testing, breakpoints and agreement metrics

## Measuring an MIC

- **Broth microdilution** (reference method, ISO 20776-1) and **agar dilution** expose the isolate to a
  two-fold series of antibiotic concentrations; the MIC is the lowest concentration without visible growth.
- The tested range is finite, so results at either end are censored (`<=` lowest, `>` highest).
- **Reproducibility**: repeat testing of the same isolate commonly differs by one doubling dilution, so an
  error of one dilution is within the noise of the measurement itself.

## Interpreting an MIC

- **Clinical breakpoints** (EUCAST or CLSI) turn an MIC into S (susceptible), I and R (resistant) for
  treatment decisions. For ciprofloxacin in Enterobacterales, EUCAST's breakpoints are S <= 0.25 mg/L and
  R > 0.5 mg/L (check the current table at <https://www.eucast.org/clinical_breakpoints>; they change).
- **Epidemiological cut-off (ECOFF)**: the upper end of the wild-type MIC distribution. Isolates above it
  have acquired some resistance mechanism, even if still clinically susceptible. EUCAST publishes MIC
  distributions and ECOFFs at <https://mic.eucast.org>.

## Agreement metrics used in MIC-prediction papers

- **Essential agreement (EA)**: predicted MIC within ±1 two-fold dilution of the measured MIC.
- **Categorical agreement (CA)**: same S/I/R category.
- **Very major error (VME)**: predicted susceptible, measured resistant (the dangerous error).
- **Major error (ME)**: predicted resistant, measured susceptible.
- Papers often compute EA after replacing censored values (`>x` by `2x`, `<=x` by `x/2`); state the
  convention whenever you report EA, because it changes the number.
- These metrics score point predictions only; a Bayesian model should also be judged on its log predictive
  density and on the calibration of its predictive intervals.
