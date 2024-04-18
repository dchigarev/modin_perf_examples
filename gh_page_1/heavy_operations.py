import numpy as np

from timeit import default_timer as timer

import pandas as pd
import modin.config as cfg

from modin.pandas.io import from_pandas, read_csv

# initialize ray
from_pandas(pd.DataFrame({"a": [1]})).modin.to_pandas()


def measure_apply():

    def users_active_at_date(date):
        return [np.random.randint(0, 1_000_000) for _ in range(10)]

    apply_func = lambda row: str(users_active_at_date(row.date())[0]).zfill(2)

    df = pd.DataFrame(
        {"date_col": pd.to_datetime(np.random.randint(0, 1_000_000, size=1_000_000))}
    )

    print("Data generated")

    t1 = timer()
    res = df["date_col"].apply(apply_func)
    pd_time = timer() - t1

    t1 = timer()
    res = from_pandas(df)["date_col"].apply(apply_func).modin.to_pandas()
    md_time = timer() - t1
    print(f"apply: {pd_time=}; {md_time=}")


def measure_merge():
    # Enabling AsyncReading is recommended for big dataframes. It consumes more memory,
    # but from_pandas() conversion works noticeably faster
    cfg.AsyncReadMode.put(True)
    df1_nrows = 80_000_000
    df1_ncols = 10
    df1_nkeys = 20_000_000

    df2_nrows = 6_000_000
    df2_ncols = 5
    df2_nkeys = 6_000_000

    df1_data = {
        "key": np.random.randint(0, df1_nkeys, size=df1_nrows),
        **{
            f"data{i}": np.random.randint(0, 1_000_000, size=df1_nrows)
            for i in range(df1_ncols - 1)
        },
    }

    df2_data = {
        "key": np.random.randint(0, df2_nkeys, size=df2_nrows),
        **{
            f"data{i}": np.random.randint(0, 1_000_000, size=df2_nrows)
            for i in range(df2_ncols - 1)
        },
    }

    df1 = pd.DataFrame(df1_data)
    df2 = pd.DataFrame(df2_data)

    t1 = timer()
    df1.merge(df2, on="key")
    pd_time = timer() - t1

    t1 = timer()
    df1 = from_pandas(df1)
    df2 = from_pandas(df2)
    df1.merge(df2, on="key").modin.to_pandas()
    md_time = timer() - t1
    print(f"merge: {pd_time=}; {md_time=}")
    cfg.AsyncReadMode.put(False)


def measure_groupby_rolling():
    NROWS = 10_000_000
    NUM_GROUPS = 250_000
    NCOLS = 10

    data = {
        "key": np.tile(np.arange(NUM_GROUPS), NROWS // NUM_GROUPS),
        **{
            f"data_col{i}": np.random.randint(0, 1_000_000, size=NROWS)
            for i in range(NCOLS - 1)
        },
    }

    df = pd.DataFrame(data)
    print("data done...")

    t1 = timer()
    df.groupby("key").rolling(10).mean()
    pd_time = timer() - t1
    print("pandas done...")

    t1 = timer()
    from_pandas(df).groupby("key").rolling(10).mean().modin.to_pandas()
    md_time = timer() - t1

    print(f"groupby: {pd_time=}; {md_time=}")


def measure_read_csv():
    NROWS = 40_000_000
    NCOLS = 10

    data = {
        f"data_col{i}": np.random.randint(0, 1_000_000, size=NROWS)
        for i in range(NCOLS)
    }

    import tempfile

    with tempfile.NamedTemporaryFile() as file:
        df = pd.DataFrame(data).to_csv(file.name)
        print("data done...")

        t1 = timer()
        pd.read_csv(file.name)
        pd_time = timer() - t1
        print("pandas done...")

        t1 = timer()
        read_csv(file.name).modin.to_pandas()
        md_time = timer() - t1

        print(f"read_csv: {pd_time=}; {md_time=}")


measure_read_csv()
measure_apply()
measure_merge()
measure_groupby_rolling()
