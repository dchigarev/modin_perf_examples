import modin.pandas as pd
import numpy as np
import modin.config as cfg

from modin.utils import execute
from timeit import default_timer as timer
import pandas

cfg.CpuCount.put(16)

def get_data(nrows, dtype):
    if dtype == int:
        return np.arange(nrows)
    elif dtype == float:
        return np.arange(nrows).astype(float)
    elif dtype == str:
        return np.array([f"value{i}" for i in range(nrows)])
    else:
        raise NotImplementedError(dtype)

pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

nrows = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]
duplicate_rate = [0, 0.1, 0.5, 0.95]
dtypes = [int, str]
use_range_part = [True, False]

columns = pandas.MultiIndex.from_product([dtypes, duplicate_rate, use_range_part], names=["dtype", "duplicate rate", "use range-part"])
result = pandas.DataFrame(index=nrows, columns=columns)

i = 0
total_its = len(nrows) * len(duplicate_rate) * len(dtypes) * len(use_range_part)

for dt in dtypes:
    for nrow in nrows:
        data = get_data(nrow, dt)
        np.random.shuffle(data)
        for dpr in duplicate_rate:
            data_c = data.copy()
            dupl_val = data_c[0]

            num_duplicates = int(dpr * nrow)
            dupl_indices = np.random.choice(np.arange(nrow), num_duplicates, replace=False)
            data_c[dupl_indices] = dupl_val

            for impl in use_range_part:
                print(f"{round((i / total_its) * 100, 2)}%")
                i += 1
                cfg.RangePartitioning.put(impl)

                sr = pd.Series(data_c)
                execute(sr)

                t1 = timer()
                # returns a list, so no need for materialization
                sr.unique()
                tm = timer() - t1
                print(nrow, dpr, dt, impl, tm)
                result.loc[nrow, (dt, dpr, impl)] = tm
                result.to_excel("unique.xlsx")
