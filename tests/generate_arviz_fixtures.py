"""Reference values from ArviZ for the rhat/ess tests (tests/diagnostics.test.js).

Regenerate with:
    uv run --no-project --python 3.12 --with arviz==0.20.0 --with "matplotlib<3.10" \
        --with "numpy<2.2" --with "scipy<1.15" --with "xarray<2025" \
        python tests/generate_arviz_fixtures.py > tests/fixtures/arviz-diagnostics.json
"""
import json, sys
import numpy as np
import arviz as az
rng = np.random.default_rng(20260922)
cases = {}
# 1. well-mixed iid normal, 4 chains x 200
cases["iid"] = rng.normal(size=(4, 200))
# 2. autocorrelated AR(1) phi=0.9, 4 chains x 300
x = np.zeros((4, 300))
for c in range(4):
    for t in range(1, 300): x[c, t] = 0.9 * x[c, t-1] + rng.normal()
cases["ar1"] = x
# 3. one chain shifted (non-converged), 4 x 200
y = rng.normal(size=(4, 200)); y[3] += 1.5
cases["shifted"] = y
# 4. heavy-tailed (Cauchy) iid, 4 x 200
cases["cauchy"] = rng.standard_cauchy(size=(4, 200))
# 5. scale mismatch: same location, one chain 3x wider (folded R-hat catches it)
s = rng.normal(size=(4, 200)); s[0] *= 3
cases["scale"] = s
out = {}
for k, v in cases.items():
    d = az.convert_to_dataset({"x": v})
    out[k] = {"chains": v.round(10).tolist(),
              "rhat": float(az.rhat(d, method="rank")["x"]),
              "ess_bulk": float(az.ess(d, method="bulk")["x"]),
              "ess_tail": float(az.ess(d, method="tail")["x"])}
    print(k, out[k]["rhat"], out[k]["ess_bulk"], out[k]["ess_tail"], file=sys.stderr)
print(json.dumps(out))
