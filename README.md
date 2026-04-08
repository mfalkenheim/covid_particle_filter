# COVID-19 Particle Filter: Estimating Campus Infection Rates

Bayesian sequential Monte Carlo estimation of latent COVID-19 infection
prevalence from mandatory testing data, with MLE parameter fitting and
backward smoothing.

---

## Background

In fall 2020, Boston University implemented one of the most rigorous
campus COVID-19 surveillance programs in the United States: mandatory
twice-weekly PCR testing of all students, isolation housing for positive
cases, and strict limits on gatherings. This produced a high-frequency
panel of positive test counts against a known testing denominator —
near-ideal conditions for Bayesian state-space estimation.

The core inferential challenge is that the *observed* positive test rate
is a noisy, partial signal of the *latent* true infection rate. A
particle filter recovers the latent state by maintaining a weighted
ensemble ("swarm") of candidate infection-rate trajectories, updating
particle weights at each observation via Bayes' rule, and resampling to
prevent particle degeneracy.

This project was developed in September 2020 as both a research exercise
and a [teaching tool for colleagues learning particle filter methods](A paticle filter estimate of COVID-19 infection rates.pdf)

---

## Model

**State variable:** `share_infected` — share of the campus population
currently infectious.

**Transition dynamics (deterministic skeleton):**

```
share_{t+1} = share_t × (1 + (R_t − 1) / serial_interval)
              + share_imported / serial_interval
```

The first term captures within-campus transmission; if R_t < 1
(as BU's mitigation achieved), infections decay. The second term seeds
new infections imported daily from off-campus exposure to Boston.

**Diffusion:** Gamma-distributed noise around the deterministic
projection. The Gamma distribution ensures non-negativity and is
standard in actuarial and epidemiological modeling for rate processes.

**Observation model:** Observed positives ~ Poisson(share_infected ×
number_tests). For the likelihood approximation, the negative binomial
(as a Gamma-Poisson mixture) accounts for overdispersion.

**Parameters estimated by MLE:**

| Parameter | MLE Estimate | Interpretation |
|---|---|---|
| `max_prior` | ~0.77% | Upper bound: share of students arriving infected |
| `r_t` | ~0.29 | Effective reproduction number on campus |
| `share_imported` | ~0.04% | Daily importation rate from Boston |
| `shape_gamma` | ~19.2 | Gamma shape; daily SD ≈ ¼ of expected value |

BU's mitigation measures achieved an R_t well below 1 — roughly
comparable to outcomes in East Asian countries with strong public health
responses at the time.

---

## Methodological note

Particle filters generalize the Kalman filter to nonlinear, non-Gaussian
state-space models. They are widely used in engineering (robot
localization), macroeconomic estimation (Fernández-Villaverde &
Rubio-Ramírez 2007), and epidemiology. The backward-smoothing pass
recovers the full posterior over the state trajectory conditional on
*all* observations, rather than just the filtering distribution
conditional on observations up to time t — generally more accurate for
retrospective analysis.

The inferential structure here — recovering a latent state from noisy
partial observations using Bayesian sequential updating — is applicable
to a broad class of measurement problems in economics and policy
analysis, including productivity measurement, labor market state
estimation, and the analysis of AI usage patterns from large-scale
behavioral data.

---

## Installation

```bash
pip install pfilter numpy pandas scipy matplotlib
```

Developed with Python 3.10+. Confirmed compatible with current `pfilter`
release.

---

## Data

The model expects a CSV file (`BU_Covid.csv`) with the following columns:

| Column | Type | Description |
|---|---|---|
| `Date` | str | Date, parseable by `pandas.to_datetime` |
| `Tests` | int | PCR tests administered |
| `Positive` | int | Positive results |
| `Positive Rate` | float | Observed positive share (Positive / Tests) |

BU published daily testing dashboards publicly during fall 2020. The
data used here were collected from those public reports.

---

## Usage

```bash
python covid_particle_filter.py
```

Or from another module:

```python
from covid_particle_filter import run_analysis, plot_results

BU_data, result = run_analysis("BU_Covid.csv")
plot_results(BU_data)
```

`run_analysis()` returns the input DataFrame augmented with `filtered`
and `smoothed` infection-rate columns, and the `scipy.OptimizeResult`
from the MLE fit.

---

## Repository structure

```
├── covid_particle_filter.py   # Main module
├── BU_Covid.csv               # Testing data (if included)
├── requirements.txt
└── README.md
```

---

## References

Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). Estimating
macroeconomic models: A likelihood approach. *The Review of Economic
Studies*, 74(4), 1059–1087.

Nordh, J. (2017). pyParticleEst: A Python framework for particle-based
estimation methods. *Journal of Statistical Software*, 78(1), 1–25.

---

## Author

Michael Falkenheim  
September 2020
