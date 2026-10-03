"""Stream one crop/country from cybench-data.zip and aggregate daily/weekly predictors to calendar dekads.
Usage: python agg.py cybench-data.zip wheat IN   ->  data/wheat_IN/{yield,crop_calendar,soil,location,crop_mask}.csv
                                                    data/wheat_IN/{fpar,ndvi,soil_moisture,meteo}_dekad.csv.gz
Dekads: days 1-10, 11-20, 21-end of month (index 0..35). prec, et0, cwb are summed; all others averaged."""
import sys, zipfile, os, time
import numpy as np, pandas as pd

Z, crop, cc = sys.argv[1], sys.argv[2], sys.argv[3]
out = f"data/{crop}_{cc}"; os.makedirs(out, exist_ok=True)
zf = zipfile.ZipFile(Z); base = f"cybench-data/{crop}/{cc}/"
for small in ["yield", "crop_calendar", "soil", "location", "crop_mask"]:
    with zf.open(f"{base}{small}_{crop}_{cc}.csv") as f, open(f"{out}/{small}.csv", "wb") as g:
        g.write(f.read())
SUM = {"prec", "et0", "cwb"}
for var in ["fpar", "ndvi", "soil_moisture", "meteo"]:
    t = time.time(); parts = []
    with zf.open(f"{base}{var}_{crop}_{cc}.csv") as f:
        for ch in pd.read_csv(f, chunksize=3_000_000):
            ch = ch.drop(columns=["crop_name"])
            d = ch["date"].astype(np.int64); m = (d // 100) % 100; dd = d % 100
            ch["year"] = (d // 10000).astype(np.int16); ch["dek"] = ((m - 1) * 3 + np.minimum((dd - 1) // 10, 2)).astype(np.int8)
            ch = ch.drop(columns=["date"]); cols = [c for c in ch.columns if c not in ("adm_id", "year", "dek")]; ch["n"] = 1
            parts.append(ch.groupby(["adm_id", "year", "dek"]).agg({**{c: "sum" for c in cols}, "n": "sum"}))
    g = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
    for c in [c for c in g.columns if c != "n"]:
        if c not in SUM:
            g[c] = g[c] / g["n"]
    g.drop(columns=["n"]).round(4).to_csv(f"{out}/{var}_dekad.csv.gz", compression="gzip")
    print(var, len(g), round(time.time() - t, 1))
