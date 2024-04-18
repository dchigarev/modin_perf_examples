#### Annotation

As of Modin 0.29.0 it is possible to perform the operations such as `groupby`, `merge`, `pivot_table`, `unique`,
`drop_duplicates`, `nunique` and `resample` using the range partitioning technique,
which gives significant performance speedup on the certain cases.

<h2 align="center">Range partitioning in Modin</h2>

Range partitioning is a type of relational database partitioning wherein the partition is based on a predefined range for a specific data field such as uniquely numbered IDs, dates or simple values like currency. A partitioning key column is assigned with a specific range, and when a data entry fits this range, it is assigned to this partition; otherwise it is placed in another partition where it fits.

Due to the flexible Modin's partitioning, which partitions data both along rows and columns, the range partitioning technique
can be applied to a specific set of operations, which benefit from it.

To enable the range partitioning technique in Modin you can specify the following environment variable:

```bash
export MODIN_RANGE_PARTITIONING=True
```

or turn it on in source code:

```python
import modin.config as cfg
 
cfg.RangePartitioning.put(True)
```

This will globally enable the technique for the operations that support it.

Since it is not always obvious when a certain operation would benefit from the range partitioning,
you can enable the respective configuration only for a certain code snippet.

```python
import modin.config as cfg

with cfg.context(RangePartitioning=True):
    df.groupby(...)  # will use the range partitioning technique

df.groupby(...)  # will not use the range partitioning technique
```

Below you can find performance measurements for some operations that benefit from using the range partitioning.
For more details on when the technique should be used refer to
[Modin's Range Partitioning](https://modin.readthedocs.io/en/latest/usage_guide/optimization_notes/range_partitioning_ops.html) page
in the documentation of Modin.

### unique

Return unique values of a Series object. The duplicate rate shows the procentage of duplicated rows in the dataset.
You can learn more about this micro-benchmark by reading its
[source code](https://github.com/dchigarev/modin_perf_examples/blob/master/gh_page_2/unique_op.py).

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/unique_16cpus.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>

Range-partitioning implementation of `unique` works better when the input data size is big (more than 5_000_000 rows) and
when the output size is also expected to be big (no more than 80% values are duplicates).

### drop_duplicates

Return a DataFrame with duplicate rows removed. The duplicate rate shows the procentage of duplicated rows in the dataset.
The subset size shows the number of columns being specified as a subset parameter for `drop_duplicates`.
You can learn more about this micro-benchmark by reading its
[source code](https://github.com/dchigarev/modin_perf_examples/blob/master/gh_page_2/drop_duplicates_op.py).

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/drop_duplicates_16cpus.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>

Range-partitioning implementation of `drop_duplicates` works better when the input data size is big (more than 5_000_000 rows)
and when the output size is also expected to be big (no more than 80% values are duplicates).

### resample

Resample time-series data and compute sum of group values. You can learn more about this micro-benchmark by reading its
[source code](https://github.com/dchigarev/modin_perf_examples/blob/master/gh_page_2/resample_sum_op.py).

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/resample_16cpus.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>

It is recommended to use the range partitioning for resampling if you're dealing with a dataframe
that has more than 5_000_000 rows and the expected output is also expected to be big (more than 500_000 rows).

#### Appendix

All performance measurements for this post were made on HP ZCentral 4R Workstation (Intel Xeon W-2245 8 cores/16 threads; 64gb RAM).

- Modin version: 0.29.0
- Pandas version: 2.2.1
- Execution engine: Ray
- Ray version: 2.9.1
- OS: Ubuntu 22.04.2 LTS
- Python: 3.9.18
