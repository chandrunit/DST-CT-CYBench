"""Anchor selection check (section 4.2): RMSE of the three-season mean and of the per-region linear trend on seasons that
precede every test year (India 2008-2012, USA 2014-2020). Uses only yields of earlier seasons for each forecast."""
import numpy as np
from lib import load, history_features

for ds, yrs in [("wheat_IN", range(2008, 2013)), ("maize_IN", range(2008, 2013)), ("maize_US", range(2014, 2021))]:
    D = load(ds); yr = D["year"]
    clim, last, trend, mean3 = history_features(D, set(yr))
    m = np.isin(yr, list(yrs))
    out = []
    for name, v in (("mean3", mean3), ("trend", trend)):
        ok = m & ~np.isnan(v)
        out.append(f"{name} {np.sqrt(np.mean((v[ok] - D['y'][ok]) ** 2)):.3f} (n={ok.sum()})")
    print(ds, "; ".join(out))
