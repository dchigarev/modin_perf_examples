import numpy as np

NROWS = 10_000_000
NUM_CODES = 1_000_000
NREFERENCES = 10_000
NCOLS = 10

print("generating data...")

data = {
    "Goods": np.tile(np.arange(NUM_CODES), NROWS // NUM_CODES),
    "Price": np.random.randint(100, 150, size=NROWS),
    **{
        f"data_col{i}": np.random.randint(0, 1_000_000, size=NROWS)
        for i in range(NCOLS - 2)
    },
}

reference_mean_prices = {
    "Goods": np.random.choice(np.arange(NUM_CODES), size=NREFERENCES, replace=False),
    "Price": np.random.randint(110, 130, size=NREFERENCES),
}

import modin.pandas as pd
import pandas

from modin.utils import execute
import modin.config as cfg

# initialize Ray
pd.DataFrame(np.arange(cfg.NPartitions.get() * cfg.MinPartitionSize.get())).to_numpy()

df = pd.DataFrame(data)
reference_mean_prices = pd.Series(
    reference_mean_prices["Price"], index=reference_mean_prices["Goods"]
).sort_index()

from timeit import default_timer as timer

execute(df)
execute(reference_mean_prices)

print("data generated")


def case1_for_loop(df, reference_prices, run_whole_loop=False):
    """Beware, this function executes very with 'run_whole_loop=True' long: ~80s on Pandas and ~900s on Modin."""
    # Task: mark ‘Goods’ with 'ones', whose mean price across different
    # branches is greater than the threshold from 'reference_prices'
    # ‘df’ shape: (10_000_000, 10)
    # Num unique goods: 1_000_000
    # ‘len(reference_prices)’: 10_000
    t0 = timer()
    df["Flag"] = 0
    iters = []
    i = 0
    for threshold_price, code in reference_prices.items():
        t1 = timer()
        mean_price = df[df["Goods"] == code]["Price"].mean()
        if mean_price > threshold_price:
            df.loc[df["Goods"] == code, "Flag"] = 1
        iters.append(timer() - t1)
        i += 1
        if not run_whole_loop and i > 10:
            break
    execute(df)
    if run_whole_loop:
        total_time = timer() - t0
        mean_iter_time = np.mean(iters)
    else:
        total_time = np.mean(iters) * len(reference_prices)
        mean_iter_time = np.mean(iters)
    return round(total_time, 2), round(mean_iter_time, 4)


def case2_groupby_mean(df, reference_mean_prices):
    # The same code, but using groupby.mean() + .apply()
    actual_mean_prices = (
        df[["Goods", "Price"]][df["Goods"].isin(reference_mean_prices.index)]
        .groupby("Goods")["Price"]
        .mean()
    )
    mask = actual_mean_prices > reference_mean_prices
    df["Flag"] = df["Goods"].apply(
        lambda code, mask: int(mask.get(code, False)), args=(mask,)
    )
    return df

def measure_cases(df, reference_mean_prices):
    t1 = timer()
    total_time, mean_iter_time = case1_for_loop(df, reference_mean_prices)

    t1 = timer()
    res = case2_groupby_mean(df, reference_mean_prices)
    # trigger execution for Modin for fair measurement
    execute(res)
    second_case_time = round(timer() - t1, 2)

    return (total_time, mean_iter_time), second_case_time

# modin
print("running test cases for modin...")
(md_first_t, md_first_m), md_second = measure_cases(df, reference_mean_prices)

# pandas
print("running test cases for pandas...")
(pd_first_t, pd_first_m), pd_second = measure_cases(df.modin.to_pandas(), reference_mean_prices.modin.to_pandas())

print(f"Modin:\n\tmean iteration time: {md_first_m}s\n\testimated case1_for_loop time: {md_first_t}s\n\tcase2_groupby_mean time: {md_second}s")
print(f"Pandas:\n\tmean iteration time: {pd_first_m}s\n\testimated case1_for_loop time: {pd_first_t}s\n\tcase2_groupby_mean time: {pd_second}s")
