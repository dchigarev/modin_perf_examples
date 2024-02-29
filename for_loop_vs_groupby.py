import numpy as np

NROWS = 10_000_000
NUM_CODES = 1_000_000
NREFERENCES = 10_000
NCOLS = 10

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

# select here which library you want to use for the measurements
import modin.pandas as pd

# import pandas as pd

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


def case1_for_loop(df, reference_prices):
    """Beware, this function executes very long: ~80s on Pandas and ~900s on Modin."""
    # Task: mark ‘Goods’ with 'ones', whose mean price across different
    # branches is greater than the threshold from 'reference_prices'
    # ‘df’ shape: (10_000_000, 10)
    # Num unique goods: 1_000_000
    # ‘len(reference_prices)’: 10_000
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
        if i > 10:
            break
    print("mean", np.mean(iters))
    return df


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


print("data done...")

t1 = timer()
res = case2_groupby_mean(df, reference_mean_prices)
# trigger execution for Modin for fair measurement
execute(res)
print("time:", timer() - t1)
