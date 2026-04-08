# -*- coding: utf-8 -*-
"""
COVID-19 Particle Filter Model
===============================

Estimates latent COVID-19 infection prevalence on a college campus from
daily mandatory testing data using a Sequential Monte Carlo (particle filter)
approach, with maximum-likelihood estimation of structural parameters and
backward smoothing.

Background
----------
In fall 2020, Boston University implemented one of the most aggressive
campus COVID-19 testing regimes in the United States: mandatory twice-weekly
testing of all students, with isolation housing for positive cases and
strict limits on social gatherings. This produced a high-frequency time
series of positive test counts against a known testing denominator — ideal
data for Bayesian state-space estimation of the underlying (unobserved)
infection rate.

The core inferential challenge is that the *observed* positive test rate is
a noisy, partial signal of the *latent* true infection rate. A particle
filter recovers the latent state by maintaining a weighted ensemble
("swarm") of candidate infection-rate trajectories, updating particle
weights at each observation using Bayes' rule, and resampling to prevent
particle degeneracy.

Model
-----
State variable
    share_infected : float in [0, 1]
        Share of the campus population currently infectious.

Transition dynamics (deterministic skeleton)::

    share_{t+1} = share_t * (1 + (R_t - 1) / serial_interval)
                  + share_imported / serial_interval

Diffusion
    Gamma-distributed noise around the deterministic projection.
    The Gamma distribution is parameterized so that its mean equals the
    projection and its shape parameter (shape_gamma) controls dispersion.
    Using a Gamma distribution ensures non-negativity and is standard
    practice in actuarial and epidemiological modeling for rate processes.

Observation model
    Observed positive tests ~ Poisson(share_infected * number_tests)
    For reweighting, the negative binomial (as a Gamma-Poisson mixture)
    provides a more robust likelihood that accounts for overdispersion.

Parameters estimated by MLE
    max_prior       : Upper bound of the Uniform prior over initial
                      infection share (students arriving on campus).
    shape_gamma     : Shape parameter of the Gamma diffusion distribution.
    r_t             : Effective reproduction number on campus.
    share_imported  : Daily rate of infection importation from off-campus.

    All parameters are transformed to the real line for unconstrained
    optimization (logit for shares/probabilities, log for positive reals).

Workflow
--------
1. Read daily testing data from CSV (columns: Date, Tests, Positive,
   Positive Rate).
2. Estimate parameters by minimizing negative log-likelihood via
   scipy.optimize.minimize.
3. Re-run filtering with fitted parameters; perform backward smoothing.
4. Plot observed positive rate alongside filtered and smoothed estimates.

Methodological note
-------------------
Particle filters generalize the Kalman filter to nonlinear, non-Gaussian
state-space models. They are widely used in engineering (robot localization),
macroeconomic estimation (Fernández-Villaverde & Rubio-Ramírez 2007), and
epidemiology. The backward-smoothing pass recovers the full posterior over
the state trajectory given all observations, rather than just the filtering
distribution conditional on observations up to time t.

This code was developed in 2020 as both a research exercise and a teaching
tool for CBO colleagues learning particle filter methods. The estimated
parameters are reported in the accompanying slides.

Dependencies
------------
    pfilter        https://pypi.org/project/pfilter/
    numpy
    pandas
    scipy
    matplotlib

Install::

    pip install pfilter numpy pandas scipy matplotlib

Usage::

    python covid_particle_filter.py

    Or from another module::

        from covid_particle_filter import run_analysis, plot_results
        BU_data, result = run_analysis("BU_Covid.csv")
        plot_results(BU_data)

Data
----
The model expects a CSV file with the following columns:

    Date          : str, parseable by pandas.to_datetime (e.g. "2020-09-01")
    Tests         : int, number of PCR tests administered on that date
    Positive      : int, number of positive results on that date
    Positive Rate : float, observed positive share (Positive / Tests)

References
----------
Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). Estimating
    macroeconomic models: A likelihood approach. The Review of Economic
    Studies, 74(4), 1059–1087.

Nordh, J. (2017). pyParticleEst: A Python framework for particle-based
    estimation methods. Journal of Statistical Software, 78(1), 1–25.

Author
------
Michael Falkenheim
Created: September 2020
"""

from math import exp, log
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pfilter import ParticleFilter, independent_sample
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import gamma, nbinom, poisson, uniform


EPS = 1e-300  # numerical floor to avoid log(0)


def _ravel(x: Any) -> np.ndarray:
    """Return a 1-D NumPy array view of *x*."""
    return np.asarray(x).reshape(-1)


def log_likelihood_fn(
    particles: np.ndarray,
    weights: np.ndarray,
    positive: int,
    **kwargs: Any,
) -> float:
    """
    Log-likelihood of observed positives under a weighted negative-binomial
    particle mixture.

    The negative binomial arises as the Gamma-Poisson mixture: if the
    expected number of positives (lambda) is Gamma-distributed across
    particles, the marginal distribution of observed positives is negative
    binomial. This accounts for overdispersion relative to a pure Poisson
    model and typically yields a better-approximated likelihood surface.

    Parameters
    ----------
    particles:
        Array of particle states, each representing a candidate value of
        share_infected at the current time step.
    weights:
        Normalized particle weights from the most recent resampling step.
    positive:
        Observed positive test count for the current period.
    **kwargs:
        Required keys:
            shape_gamma   : float  — Gamma shape parameter.
            number_tests  : int    — Tests administered in current period.

    Returns
    -------
    float
        Log-likelihood contribution for the current observation.
    """
    shape_binom = kwargs["shape_gamma"]
    projected = _ravel(dynamics_fn(particles, **kwargs)) * kwargs["number_tests"]
    p = shape_binom / (projected + shape_binom)  # NB success probability
    pmf_vals = nbinom.pmf(positive, shape_binom, p)
    mixture_like = np.dot(_ravel(weights), pmf_vals)
    return float(np.log(np.maximum(mixture_like, EPS)))


def weight_fn(projected: np.ndarray, observed: int, **kwargs: Any) -> np.ndarray:
    """
    Compute particle weights from Poisson likelihood of observed test count.

    Parameters
    ----------
    projected:
        Expected positive counts under each particle's infection rate.
    observed:
        Observed positive test count for the current period.

    Returns
    -------
    np.ndarray
        1-D array of unnormalized particle likelihoods.
    """
    return poisson.pmf(observed, projected).ravel()


def observe_fn(share_infected: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Observation function: map latent infection share to expected positive tests.

    Parameters
    ----------
    share_infected:
        Array of particle infection-share values.
    **kwargs:
        Required key: number_tests : int

    Returns
    -------
    np.ndarray
        Expected positive test counts for each particle.
    """
    return share_infected * kwargs["number_tests"]


def dynamics_fn(share_infected: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Deterministic state transition for campus infection dynamics.

    Implements a discrete-time SIS-like model::

        share_{t+1} = share_t * (1 + (R_t - 1) / serial_interval)
                      + share_imported / serial_interval

    The first term captures within-campus transmission: if R_t > 1,
    infections grow; if R_t < 1 (as BU's mitigation achieved), they decay.
    The second term seeds new infections imported daily from off-campus
    (exposure to the broader Boston population).

    Parameters
    ----------
    share_infected:
        Current particle infection-share values.
    **kwargs:
        Required keys:
            r_t             : float — Effective reproduction number.
            serial_interval : float — Mean serial interval in days.
            share_imported  : float — Daily off-campus importation rate.

    Returns
    -------
    np.ndarray
        Projected infection shares for the next period.
    """
    return (
        share_infected * (1 + (kwargs["r_t"] - 1) / kwargs["serial_interval"])
        + kwargs["share_imported"] / kwargs["serial_interval"]
    )


def diffusion_fn(projected_infected: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Gamma diffusion noise around the deterministic projection.

    Draws new particle positions from a Gamma distribution whose mean
    equals the deterministic projection and whose variance is controlled
    by shape_gamma. Higher shape_gamma implies tighter dispersion around
    the projection.

    The Gamma distribution is the natural choice for a non-negative rate
    process: it ensures share_infected remains positive and is conjugate
    to the Poisson observation model.

    Parameters
    ----------
    projected_infected:
        Deterministic projections from dynamics_fn.
    **kwargs:
        Required key: shape_gamma : float

    Returns
    -------
    np.ndarray
        Noisy particle positions, same shape as projected_infected.
    """
    projected = np.asarray(projected_infected)
    return gamma.rvs(
        kwargs["shape_gamma"],
        scale=projected / kwargs["shape_gamma"],
        size=projected.shape,
    )


def diffusion_prob_fn(
    projected_infected: np.ndarray,
    infected: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    Normalized backward-smoothing weights under the Gamma transition density.

    Used in the backward pass to re-weight particles at time t given the
    smoothed state estimate at time t+1. For each particle at time t,
    computes the probability of transitioning to the smoothed value at
    t+1 under the Gamma diffusion kernel.

    Parameters
    ----------
    projected_infected:
        Forward projections from time-t particles (via dynamics_fn).
    infected:
        Target infection-share values at time t+1 (smoothed estimate
        broadcast to particle shape).
    **kwargs:
        Required key: shape_gamma : float

    Returns
    -------
    np.ndarray
        Normalized weights summing to 1. Falls back to uniform weights
        if the Gamma density integrates to zero (degenerate case).
    """
    projected = _ravel(projected_infected)
    infected = _ravel(infected)
    w = gamma.pdf(
        infected,
        kwargs["shape_gamma"],
        scale=projected / kwargs["shape_gamma"],
    )
    s = w.sum()
    if s <= 0:
        return np.full_like(w, 1.0 / w.size, dtype=float)
    return w / s


def get_filtered_smoothed(
    all_particles: list[np.ndarray],
    all_weights: list[np.ndarray],
    dynamics_fn: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute filtered and backward-smoothed infection-rate estimates.

    The filtering estimate at time t is the weighted mean of particles
    given observations up to time t. The smoothing estimate incorporates
    all observations (past and future) via a backward pass, and is
    generally more accurate.

    Parameters
    ----------
    all_particles:
        List of particle arrays stored during the forward filtering pass,
        one entry per time period.
    all_weights:
        Corresponding list of normalized particle weight arrays.
    dynamics_fn:
        State transition function (passed through for backward weighting).
    **kwargs:
        Model parameters forwarded to diffusion_prob_fn.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        filtered : weighted-mean infection share at each time step,
                   conditioning only on past observations.
        smoothed : weighted-mean infection share at each time step,
                   conditioning on all observations.
    """
    n_periods = len(all_particles)
    filtered = np.zeros(n_periods, dtype=float)
    smoothed = np.zeros(n_periods, dtype=float)

    # Initialise terminal condition: filter == smoother at last period
    filtered[-1] = np.dot(_ravel(all_weights[-1]), _ravel(all_particles[-1]))
    smoothed[-1] = filtered[-1]
    last_smoothed = smoothed[-1]

    for t in range(n_periods - 2, -1, -1):
        p_t = _ravel(all_particles[t])
        w_t = _ravel(all_weights[t])

        filtered[t] = np.dot(w_t, p_t)

        projected = _ravel(dynamics_fn(p_t, **kwargs))
        smooth_w = diffusion_prob_fn(
            projected,
            np.full_like(projected, last_smoothed),
            **kwargs,
        )
        smoothed[t] = np.dot(smooth_w, p_t)
        last_smoothed = smoothed[t]

    return filtered, smoothed


def make_negative_ll(x: np.ndarray, args: dict) -> float:
    """
    Objective function: negative log-likelihood over all observation dates.

    Parameters are transformed to the real line for unconstrained
    optimization::

        x[0] : logit(max_prior)      — initial infection share upper bound
        x[1] : log(shape_gamma)      — Gamma diffusion shape
        x[2] : log(r_t)              — effective reproduction number
        x[3] : logit(share_imported) — daily off-campus importation share

    A fixed random seed is set before each filter run so that the
    stochastic likelihood approximation is deterministic across optimizer
    calls, which is necessary for gradient-based methods.

    Parameters
    ----------
    x:
        Transformed parameter vector (length 4).
    args:
        Dictionary with keys: random_number_seed, serial_interval,
        tests (np.ndarray), positives (np.ndarray), n_particles_opt (int).

    Returns
    -------
    float
        Negative log-likelihood (to be minimized).
    """
    kwargs: dict[str, Any] = {
        "r_t": exp(x[2]),
        "serial_interval": args["serial_interval"],
        "share_imported": exp(x[3]) / (1 + exp(x[3])),
        "number_tests": 2000,
        "max_prior": exp(x[0]) / (1 + exp(x[0])),
        "shape_gamma": exp(x[1]),
    }

    np.random.seed(args["random_number_seed"])

    pf = ParticleFilter(
        prior_fn=independent_sample([uniform(0, kwargs["max_prior"]).rvs]),
        observe_fn=observe_fn,
        n_particles=args.get("n_particles_opt", 120),
        dynamics_fn=dynamics_fn,
        noise_fn=diffusion_fn,
        weight_fn=weight_fn,
    )

    tests = args["tests"]
    positives = args["positives"]

    nll = 0.0
    for i in range(tests.size):
        kwargs["number_tests"] = int(tests[i])
        y = int(positives[i])
        nll -= log_likelihood_fn(pf.original_particles, pf.weights, y, **kwargs)
        if pf.weights.ndim > 1:
            pf.weights = pf.weights.ravel()
        pf.update(np.asarray(y), **kwargs)

    return float(nll)


def run_analysis(
    csv_path: str = "BU_Covid.csv",
    random_seed: int = 17,
    serial_interval: float = 6.0,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """
    Run the full model: parameter estimation, filtering, and smoothing.

    Optimization proceeds in two stages:
    1. Fast initial fit with 120 particles (maxiter=150) to find a good
       starting point.
    2. Refinement fit with 300 particles from the stage-1 solution.

    The Hessian inverse from scipy's L-BFGS-B optimizer provides an
    approximate covariance matrix for the transformed parameters, from
    which standard errors can be derived (though bootstrap methods would
    be more reliable for a formal inference exercise).

    Parameters
    ----------
    csv_path:
        Path to the BU testing data CSV file.
    random_seed:
        Random seed fixed before each particle filter run to ensure
        deterministic likelihood evaluations during optimization.
    serial_interval:
        Mean serial interval (days) assumed for the virus. Default 6.0
        is consistent with early COVID-19 estimates.

    Returns
    -------
    tuple[pd.DataFrame, OptimizeResult]
        BU_data   : Input data augmented with 'filtered' and 'smoothed'
                    infection-rate columns.
        x_optimal : scipy OptimizeResult from the refinement fit,
                    including fitted parameters and approximate Hessian.
    """
    BU_data = pd.read_csv(csv_path)
    BU_data["Date"] = pd.to_datetime(BU_data["Date"])

    # Initial parameter values (transformed):
    #   max_prior     ~ 0.25%  ->  logit(0.0025)
    #   shape_gamma   ~ 7.5    ->  log(7.5)
    #   r_t           ~ 0.9    ->  log(0.9)
    #   share_imported~ 0.03%  ->  logit(0.0003)
    x0 = np.array(
        [
            log(0.0025 / (1 - 0.0025)),
            log(7.5),
            log(0.9),
            log(0.0003 / (1 - 0.0003)),
        ],
        dtype=float,
    )

    args: dict[str, Any] = {
        "data": BU_data,
        "random_number_seed": random_seed,
        "serial_interval": serial_interval,
        "tests": BU_data["Tests"].to_numpy(),
        "positives": BU_data["Positive"].to_numpy(),
        "n_particles_opt": 120,
    }

    # Stage 1: fast fit
    result_stage1 = minimize(
        make_negative_ll,
        x0,
        args=(args,),
        options={"maxiter": 150},
    )

    # Stage 2: refinement with more particles
    args["n_particles_opt"] = 300
    x_optimal = minimize(
        make_negative_ll,
        result_stage1.x,
        args=(args,),
    )

    x_opt = x_optimal.x
    estimates = {
        "max_prior":      exp(x_opt[0]) / (1 + exp(x_opt[0])),
        "shape_gamma":    exp(x_opt[1]),
        "r_t":            exp(x_opt[2]),
        "share_imported": exp(x_opt[3]) / (1 + exp(x_opt[3])),
    }

    # Re-run filter with fitted parameters to collect particle history
    kwargs: dict[str, Any] = {
        "r_t":             estimates["r_t"],
        "serial_interval": serial_interval,
        "share_imported":  estimates["share_imported"],
        "number_tests":    2000,
        "max_prior":       estimates["max_prior"],
        "shape_gamma":     estimates["shape_gamma"],
    }

    pf = ParticleFilter(
        prior_fn=independent_sample([uniform(0, kwargs["max_prior"]).rvs]),
        observe_fn=observe_fn,
        n_particles=300,
        dynamics_fn=dynamics_fn,
        noise_fn=diffusion_fn,
        weight_fn=weight_fn,
    )

    all_weights: list[np.ndarray] = []
    all_particles: list[np.ndarray] = []

    for i in range(len(BU_data)):
        kwargs["number_tests"] = int(BU_data.iloc[i]["Tests"])
        if pf.weights.ndim > 1:
            pf.weights = pf.weights.ravel()
        pf.update(np.asarray(BU_data.iloc[i]["Positive"]), **kwargs)
        all_weights.append(pf.weights.copy())
        all_particles.append(pf.original_particles.copy())

    filtered, smoothed = get_filtered_smoothed(
        all_particles, all_weights, dynamics_fn, **kwargs
    )
    BU_data["filtered"] = filtered
    BU_data["smoothed"] = smoothed

    return BU_data, x_optimal


def plot_results(BU_data: pd.DataFrame) -> None:
    """
    Plot observed positive rate alongside filtered and smoothed estimates.

    Parameters
    ----------
    BU_data:
        DataFrame returned by run_analysis, containing Date,
        Positive Rate, filtered, and smoothed columns.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        BU_data["Date"],
        BU_data["Positive Rate"],
        label="Observed Positive Rate",
        linewidth=1.8,
    )
    ax.plot(
        BU_data["Date"],
        BU_data["filtered"],
        linestyle="-.",
        label="Filtered Estimate",
        linewidth=1.6,
    )
    ax.plot(
        BU_data["Date"],
        BU_data["smoothed"],
        linestyle="--",
        label="Smoothed Estimate",
        linewidth=1.6,
    )

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xticks(rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of Positive Tests")
    ax.set_title("COVID-19 Infection Estimates: Boston University Fall 2020")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    BU_data, x_optimal = run_analysis()
    plot_results(BU_data)
