#### Annotation
Discover the power of Modin in overcoming the performance bottlenecks of Pandas. We'll walk you through how a simple switch can reduce your waiting times and make your workflow smooth and speedy, all without leaving the comfort of Pandas behind.

<h2 align="center">Breaking a single core limitation of Pandas with Modin</h2>

Imagine you are writing a data-preprocessing notebook using Pandas, and the total execution time of 4min+ doesn’t satisfy you. You start measuring individual stages of the workload and see a picture like this: 97% of the whole workload’s time is taken by a single operation, such as `df.groupby()`, `df.apply()` or `df.merge()`


<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img1_customer_segmentation_pandas.jpg" style="display: block;margin-left: auto;margin-right: auto; width:50%; padding: 0; margin: 0"></img>
###### Simplified version of 'Customer segmentation' notebook: https://github.com/dchigarev/modin_perf_examples/blob/master/customer_segmentation_simplified.py<br>Original notebook: https://www.kaggle.com/code/fabiendaniel/customer-segmentation/notebook

You are struggling to speed up the Pandas operation due to its single-core limitation and the complexity of parallelizing tasks like groupby or merge on your own. Modin could be a solution here, offering a drop-in replacement for Pandas and efficient parallel implementations of its API. It’s enough only to change your import statement to start using Modin:
```python
# import pandas as pd
import modin.pandas as pd
 
# your workload without any changes
```

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img2_customer_segmentation_pandas_vs_modin.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>
###### Measurements for the ‘customer segmentation’ workload after changing the import statement from Pandas to Modin

In comparison with other distributed dataframe libraries, Modin claims to be 99% Pandas compatible, meaning that you don’t need to modify your Pandas code at all to apply Modin. Note how Modin has been able to process the groupby even though it used a complex custom aggregation function that was written specifically for Pandas *refer the code of the agg function, maybe hide it under a spoiler*.

Although Modin promises speedup after changing only the import statement, it is not a magic pill that makes everything faster. As a practice-based observation, Modin’s implementation for a certain method works faster if the operation takes longer than 5 seconds in Pandas. That would be an explanation for why we see slowdowns in certain parts of the workflow here after applying Modin.

If you find that Modin performs worse in certain parts of your workflow, just like in our example, then you can engage Modin only in the parts where it is beneficial, in our case, it is the section with the heavy `groupby.apply()` call. By simply wrapping it into Modin’s objects you get a 7x speed-up for this groupby and not lose time in other parts of your workflow where Pandas works better:

```diff
# Do not change you import statement
import pandas as pd
...
+from modin.pandas.io import from_pandas
+
-grp = df_initial.groupby(
+grp = from_pandas(df_initial).groupby(
     ["CustomerID", "StockCode"], as_index=False
 )
-df_cleaned = grp.apply(groupby_filtering)
+df_cleaned = grp.apply(groupby_filtering).to_pandas()
```

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img3_customer_segmentation_mixed_pandas_modin.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>
###### Measurements for the ‘customer segmentation’ workload Pandas vs Modin vs Modin + Pandas

Modin has efficient parallel implementations for most of the Pandas methods, covering all the variety of parameters they can take. Here is a little showcase demonstrating how Modin deals with the heaviest operations in pandas:

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img4_heavy_operations.jpg" style="display: block;margin-left: auto;margin-right: auto; width:50%; padding: 0; margin: 0"></img>
###### Link to the full script with the source code: https://github.com/dchigarev/modin_perf_examples/blob/master/heavy_operations.py

Even if Modin doesn’t have a parallel implementation for a specific method, it defaults to using Pandas implementation for that method, issuing a warning in the process. This ensures that the workflow proceeds without interruption.

### Speeding up a sequence of heavy Pandas operations
Modin can also be seamlessly integrated into a sequence of intensive operations within your workflow by simply converting the required dataframes to Modin. As already mentioned, Modin works far beyond `groupby()`, `apply()` and other popular Pandas methods, so any intermediate pandas-exotic calls between heavy operations wouldn’t be a problem for Modin:

```python
from modin.pandas.io import from_pandas

# Heavy piece of code, converting to Modin at the beginning
filtered = from_pandas(df).query("Price < 100_000 & BuyCount > 0")
mean_per_branch = filtered.groupby("Branch").apply(cals_stats)
mean_per_city = filtered.groupby("City").apply(calc_stats)

merged = filtered.merge(mean_per_branch, on="Branch")
                 .merge(mean_per_city, on="City")
                 .to_pandas() # Converting back to Pandas at the end
                              # of the heavy block
```
###### Link to the full script with the source code: https://github.com/dchigarev/modin_perf_examples/blob/master/sequence_of_heavy_ops.py
<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img5_sequence_of_heavy_ops.jpg" style="display: block;margin-left: auto;margin-right: auto; width:80%; padding: 0; margin: 0"></img>

### When not to use Modin

Certain operations benefit more from parallelization than others. Due to its distributed nature, Modin might not outperform Pandas in speed for some methods under specific circumstances.

For example, it was already mentioned that applying Modin for a certain operation is recommended only if it takes more than 5 seconds on Pandas. The explanation is simple – the overhead of data distribution and parallel execution is only justifiable when working on substantial tasks. One of the common Modin’s anti-patterns we see in user’s code is using it within for-loops:
```python
# Task: mark ‘Goods’ with 'ones', whose mean price across different
# branches is greater than the threshold from 'reference_prices'
# ‘df’ shape: (10_000_000, 10)
# Num unique goods: 1_000_000
# ‘len(reference_prices)’: 10_000
df["Flag"] = 0
for threshold_price, code in reference_prices.items():
    mean_price = df[df["Goods"] == code]["Price"].mean()
    flag_value = mean_price > threshold_price
    if mean_price > threshold_price:
        df.loc[df["Goods"] == code, "Flag"] = 1
```
###### Link to the full script with the source code: https://github.com/dchigarev/modin_perf_examples/blob/master/for_loop_vs_groupby.py

In the example above, the whole loop takes about 80 seconds, however each iteration is simple and takes less than a second on Pandas, what makes this loop slow - is the number of iterations.

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img6_modin_in_a_for_loop.jpg" style="display: block;margin-left: auto;margin-right: auto; width:50%; padding: 0; margin: 0"></img>

Trying to apply Modin would be a mistake here. Modin can’t magically parallelize python’s for-loops, iterations are still executed sequentially. Moreover, the overhead of sequentially distributing each tiny iteration would eat up all the profit and rather slow-down the whole loop. Instead, you would want to rewrite this loop using Pandas API and apply Modin afterwards:

```python
# The same code, but using groupby.mean() + .apply()
actual_mean_prices = (
    df[["Goods", "Price"]][
        df["Goods"].isin(reference_mean_prices.index)
    ]
    .groupby("Goods")["Price"]
    .mean()
)
mask = actual_mean_prices > reference_mean_prices
df["Flag"] = df["Goods"].apply(
    lambda code, mask: int(mask.get(code, False)), args=(mask,)
)
```

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img7_modin_for_loop_rewritten.jpg" style="display: block;margin-left: auto;margin-right: auto; width:50%; padding: 0; margin: 0"></img>

It is important to note that Modin was designed to efficiently process heavy tasks, rather than a big number of small ones. Modin is under active development and targets itself for 1.0 release to work ‘not worse than Pandas’ in all variety of cases that are nowadays considered to be ‘anti-patterns’ for Modin.

But as for now, it is advised to initially introduce Modin only into parts of your workflow where Pandas is known to underperform. Consider a complete switch from Pandas to Modin after you have eliminated such Modin antipatterns from your code and still find that Pandas struggles with every operation of the workflow.

### Minimal hardware requirements
Modin achieves speed up by distributing computations over CPU cores. It is recommended to have a configuration with at least 4-cores/8-threads CPU and 32GB of RAM to see a noticeable speedup. Note that the popular free environment for running Python notebooks, Google Colab, has only a 1-core/2-threads CPU, which is not enough to use Modin.

### How Modin works underneath
Modin starts with distributing the input data – it splits the data into small portions, called partitions, along both axis: rows and columns. Each partition is a small Pandas DataFrame that is stored in an immutable shared storage.

<img src="https://github.com/dchigarev/modin_perf_examples/raw/master/docs/imgs/img8_modin_arch.jpg" style="display: block;margin-left: auto;margin-right: auto; width:60%; padding: 0; margin: 0"></img>

Then when an operation is invoked, different worker processes fetch a subset of partitions and apply an operation to each partition in parallel, writing the result back to the storage.

Modin's architecture is flexible, allowing to utilize various implementations of shared storage and execution engines that run kernels. At the moment, there are four different engines presented in Modin:

-	[Ray](https://www.ray.io) + [Plasma storage](https://ray-project.github.io/2017/08/08/plasma-in-memory-object-store.html) / in-process memory
-	[Dask](https://www.dask.org/) + in-process memory
-	[MPI](https://mpi4py.readthedocs.io/en/stable/) (via [Unidist](https://github.com/modin-project/unidist)) + Shared storage / in-process memory
-	[PyHDK](https://github.com/intel-ai/hdk) with its own Arrow-based storage (natively supports SQL queries)

You can select the execution engine by simply specifying it before the first usage of Modin via configuration variable, no other changes are necessary:
```python
import modin.config as cfg

cfg.Engine.put("Ray") # will use Ray
cfg.Engine.put("Dask") # will use Dask
# ...
```
Ray execution is considered to be the most developed one and is recommended to be used by default.

If you are familiar with Ray, you can continue using all its infrastructure tools with Modin, like [`ray timeline`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.timeline.html) profiling or [`ray dashboard`](https://docs.ray.io/en/latest/ray-observability/getting-started.html). Moreover, if you already have a configured Ray cluster, you can [easily use it](https://modin.readthedocs.io/en/stable/getting_started/using_modin/using_modin_cluster.html#using-modin-in-a-cluster) to distribute your Pandas computations with Modin.

You can learn more about Modin’s integration with Ray in [Modin’s documentation](https://modin.readthedocs.io/en/stable/). 

#### Appendix
All performance measurements for this article were made on HP ZCentral 4R Workstation (Intel Xeon W-2245 8 cores/16 threads; 64gb RAM)
- Modin version: 0.28.0
- Pandas version: 2.2.1
- Execution engine: Ray
- Ray version: 2.9.1
- OS: Ubuntu 22.04.2 LTS
