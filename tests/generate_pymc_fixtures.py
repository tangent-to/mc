#!/usr/bin/env python3
"""Generate stored PyMC posterior fixtures for tests/pymc-posterior.test.js.

Cross-validation against a real PyMC run, rather than against the
data-generating parameters. Run this to regenerate:

    uv run --with pymc --with numpy python3 tests/generate_pymc_fixtures.py

The output is committed so the test runs without a Python toolchain; PyMC is
heavy and CI has no Python. Regenerate only deliberately — a fixture change is
a change of reference, and should be reviewed as one.
"""
import json
import os
import warnings

import numpy as np
import pymc as pm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "pymc-posteriors.json")

DRAWS, TUNE, CHAINS, SEED = 4000, 2000, 4, 20260901


def linear_regression():
    """Multiple regression. The workhorse case, and the one with enough free
    parameters that the finite-difference penalty is visible."""
    rng = np.random.default_rng(11)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    beta = np.array([0.5 + 0.4 * j for j in range(p)])
    sigma = 0.7
    y = X @ beta + sigma * rng.normal(size=n)

    with pm.Model():
        b = pm.Normal("beta", mu=0, sigma=5, shape=p)
        s = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("y", mu=pm.math.dot(X, b), sigma=s, observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                          random_seed=SEED, progressbar=False,
                          compute_convergence_checks=False)

    post = idata.posterior
    bs = post["beta"].values.reshape(-1, p)
    ss = post["sigma"].values.reshape(-1)
    return {
        "X": X.tolist(), "y": y.tolist(),
        "true": {"beta": beta.tolist(), "sigma": sigma},
        "pymc": {
            "beta_mean": bs.mean(axis=0).tolist(),
            "beta_sd": bs.std(axis=0).tolist(),
            "sigma_mean": float(ss.mean()),
            "sigma_sd": float(ss.std()),
        },
    }


def eight_schools():
    """Non-centred eight schools: the standard hierarchical funnel, and a real
    test of whether a sampler explores a difficult geometry rather than just a
    well-conditioned Gaussian."""
    y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    sd = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=10)
        tau = pm.HalfNormal("tau", sigma=10)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(y))
        pm.Normal("obs", mu=theta, sigma=sd, observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                          random_seed=SEED, progressbar=False,
                          compute_convergence_checks=False, target_accept=0.95)

    post = idata.posterior
    return {
        "y": y, "sd": sd,
        "pymc": {
            "mu_mean": float(post["mu"].values.mean()),
            "mu_sd": float(post["mu"].values.std()),
            "tau_mean": float(post["tau"].values.mean()),
            "tau_sd": float(post["tau"].values.std()),
        },
    }


def main():
    out = {
        "_meta": {
            "pymc": pm.__version__,
            "draws": DRAWS, "tune": TUNE, "chains": CHAINS, "seed": SEED,
        },
        "linear_regression": linear_regression(),
        "eight_schools": eight_schools(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}  (PyMC {pm.__version__}, {DRAWS} draws x {CHAINS} chains)")


if __name__ == "__main__":
    main()
