import modin.pandas as pd
import numpy as np
import modin.config as cfg

from modin.utils import execute
from timeit import default_timer as timer
import pandas

cfg.CpuCount.put(16)

pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

nrows = [1_000_000, 5_000_000, 10_000_000, 25_000_000]
duplicate_rate = [0, 0.1, 0.5, 0.95]
subset = [["col0"], ["col1", "col2", "col3", "col4"], None]
ncols = 15
use_range_part = [True, False]

columns = pandas.MultiIndex.from_product(
    [
        [len(sbs) if sbs is not None else ncols for sbs in subset],
        duplicate_rate,
        use_range_part
    ],
    names=["subset size", "duplicate rate", "use range-part"]
)
result = pandas.DataFrame(index=nrows, columns=columns)

i = 0
total_its = len(nrows) * len(duplicate_rate) * len(subset) * len(use_range_part)

for sbs in subset:
    for nrow in nrows:
        data = {f"col{i}": np.arange(nrow) for i in range(ncols)}
        pandas_df = pandas.DataFrame(data)

        for dpr in duplicate_rate:
            pandas_df_c = pandas_df.copy()
            dupl_val = pandas_df_c.iloc[0]

            num_duplicates = int(dpr * nrow)
            dupl_indices = np.random.choice(np.arange(nrow), num_duplicates, replace=False)
            pandas_df_c.iloc[dupl_indices] = dupl_val

            for impl in use_range_part:
                print(f"{round((i / total_its) * 100, 2)}%")
                i += 1
                cfg.RangePartitioning.put(impl)

                md_df = pd.DataFrame(pandas_df_c)
                execute(md_df)

                t1 = timer()
                res = md_df.drop_duplicates(subset=sbs)
                execute(res)
                tm = timer() - t1

                sbs_s = len(sbs) if sbs is not None else ncols
                print("len()", res.shape, nrow, dpr, sbs_s, impl, tm)
                result.loc[nrow, (sbs_s, dpr, impl)] = tm
                result.to_excel("drop_dupl.xlsx")
