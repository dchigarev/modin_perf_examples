import pandas as pd
import numpy as np

from timeit import default_timer as timer

NROWS = 30_000_000
NUM_BRANCHES = 500_000
NUM_CITIES = 500
NCOLS = 10

print("generating data...")

data = {
    "Branch": np.tile(np.arange(NUM_BRANCHES), NROWS // NUM_BRANCHES),
    "City": np.tile([f"City{i}" for i in range(NUM_CITIES)], NROWS // NUM_CITIES),
    "Price": np.random.randint(50_000, 110_000, size=NROWS),
    "BuyCount": np.random.randint(0, 100, size=NROWS),
    **{
        f"data_col{i}": np.random.randint(0, 1_000_000, size=NROWS)
        for i in range(NCOLS - 4)
    },
}

df = pd.DataFrame(data)


def calc_stats(df):
    # mean of normalized values
    df = df.select_dtypes("number")
    return (df / df.sum()).mean()


from modin.pandas.io import from_pandas

# initialize Ray
from_pandas(pd.DataFrame({"a": [1]}))

print("data generated...")

def pure_pandas(df):
    filtered = df.query("Price < 100_000 & BuyCount > 0")
    stats_per_branch = filtered.groupby("Branch").apply(
        calc_stats, include_groups=False
    )
    stats_per_city = filtered.groupby("City").apply(calc_stats, include_groups=False)

    merged = filtered.merge(stats_per_branch, on="Branch").merge(
        stats_per_city, on="City"
    )


def modin_one_conversion(df):
    filtered = from_pandas(df).query("Price < 100_000 & BuyCount > 0")
    stats_per_branch = filtered.groupby("Branch").apply(
        calc_stats, include_groups=False
    )
    stats_per_city = filtered.groupby("City").apply(calc_stats, include_groups=False)

    merged = filtered.merge(stats_per_branch, on="Branch").merge(
        stats_per_city, on="City"
    )
    merged = merged.modin.to_pandas()


def modin_several_conversions(df):
    filtered = from_pandas(df).query("Price < 100_000 & BuyCount > 0").modin.to_pandas()
    stats_per_branch = (
        from_pandas(filtered).groupby("Branch").apply(calc_stats, include_groups=False)
    )
    stats_per_city = (
        from_pandas(filtered).groupby("City").apply(calc_stats, include_groups=False)
    )

    merged = from_pandas(filtered).merge(stats_per_branch, on="Branch").merge(
        stats_per_city, on="City"
    )
    merged = merged.modin.to_pandas()

print("running test cases for modin...")

t0 = timer()
modin_one_conversion(df)
print("modin one conversion:", timer() - t0)

t0 = timer()
modin_several_conversions(df)
print("modin several conversions:", timer() - t0)

print("running test cases for pandas...")

t0 = timer()
pure_pandas(df)
print("pure pandas:", timer() - t0)
