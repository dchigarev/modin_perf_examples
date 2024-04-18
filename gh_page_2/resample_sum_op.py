import pandas
import numpy as np
import modin.pandas as pd
import modin.config as cfg

from timeit import default_timer as timer

from modin.utils import execute

cfg.CpuCount.put(16)

nrows = [1_000_000, 5_000_000, 10_000_000]
ncols = [5, 33]
rules = [
    "500ms", # doubles nrows
    "30s", # decreases nrows in 30 times
    "5min", # decreases nrows in 300
]
use_rparts = [True, False]

cols = pandas.MultiIndex.from_product([rules, ncols, use_rparts], names=["rule", "ncols", "USE RANGE PART"])
rres = pandas.DataFrame(index=nrows, columns=cols)

total_nits = len(nrows) * len(ncols) * len(rules) * len(use_rparts)
i = 0

for nrow in nrows:
    for ncol in ncols:
        index = pandas.date_range("31/12/2000", periods=nrow, freq="s")
        data = {f"col{i}": np.arange(nrow) for i in range(ncol)}
        pd_df = pandas.DataFrame(data, index=index)
        for rule in rules:
            for rparts in use_rparts:
                print(f"{round((i / total_nits) * 100, 2)}%")
                i += 1
                cfg.RangePartitioning.put(rparts)

                df = pd.DataFrame(data, index=index)
                execute(df)

                t1 = timer()
                res = df.resample(rule).sum()
                execute(res)
                ts = timer() - t1
                print(nrow, ncol, rule, rparts, ts)

                rres.loc[nrow, (rule, ncol, rparts)] = ts
                rres.to_excel("resample.xlsx")
