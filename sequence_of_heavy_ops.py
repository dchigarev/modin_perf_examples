import pandas as pd
import numpy as np

from timeit import default_timer as timer

NROWS = 30_000_000
NUM_BRANCHES = 500_000
NUM_CITIES = 500
NCOLS = 10

data = {
    "Branch": np.tile(np.arange(NUM_BRANCHES), NROWS // NUM_BRANCHES),
    "City": np.tile([f"City{i}" for i in range(NUM_CITIES)], NROWS // NUM_CITIES),
    "Price": np.random.randint(50_000, 110_000, size=NROWS),
    "BuyCount": np.random.randint(0, 100, size=NROWS),
    **{f"data_col{i}": np.random.randint(0, 1_000_000, size=NROWS) for i in range(NCOLS - 4)}
}

df = pd.DataFrame(data)

def from_pandas(df):
    return df

def to_pandas(df):
    return df

def calc_stats(df):
    # mean of normalized values
    df = df.select_dtypes("number")
    return (df / df.sum()).mean()


import modin.config as cfg
from modin.pandas.io import from_pandas, to_pandas
from_pandas(pd.DataFrame({"a": [1]}))

t0 = timer()

filtered = from_pandas(df).query("Price < 100_000 & BuyCount > 0")
stats_per_branch = from_pandas(filtered).groupby("Branch").apply(calc_stats, include_groups=False)
stats_per_city = from_pandas(filtered).groupby("City").apply(calc_stats, include_groups=False)

merged = filtered.merge(stats_per_branch, on="Branch").merge(stats_per_city, on="City")
merged = to_pandas(merged)

print(timer() - t0)
