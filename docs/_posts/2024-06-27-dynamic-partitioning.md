---
layout: post
title: "Dynamic Partitioning in Modin"
categories: misc
permalink: /gh_page_2.html
author: Kirill Suvorov
---

#### Annotation

Modin improves work with partitions in the Map and TreeReduce operators in the 0.30.0 release for wide Dataframes.

## Dynamic Partitioning

As you know, Modin splits a DataFrame into several partitions to parallelize execution. By default, Modin uses the CPU count for this purpose. When an operation is executed, the required partitions are sent to remote tasks and executed in parallel. To achieve optimal performance, it’s essential to fully utilize all available CPUs. Therefore, Modin splits the DataFrame into parts along both axes, resulting in an N × N partition grid (where N represents the CPU count).

This approach works well, especially for axis operations. In such cases, we concatenate some partitions into virtual partitions and distribute them to separate remote tasks. As a result, we achieve an optimal number of remote tasks equal to the CPU count. However, there are situations where this method is less effective. For example, the Ray engine has difficulty handling too many remote tasks. This results in a significant slowdown compared to running a small number of tasks but with a large amount of data on each. So, for simple operations such as apply, using this partitioning strategy can lead to performance problems due to too many remote tasks. 

To address this problem, a new approach called “Dynamic Partitioning” has been implemented in Modin 0.30.0. The main idea behind Dynamic Partitioning is to merge block partitions into virtual partitions, reducing the overall number of remote tasks.

## Boosted DataFrame operations

In Modin 0.30.0, dynamic partitioning is applied to the [Map](https://modin.readthedocs.io/en/latest/flow/modin/core/dataframe/algebra.html#map-operator) and [TreeReduce](https://modin.readthedocs.io/en/latest/flow/modin/core/dataframe/algebra.html#map-operator) operators.

The Map operator is used for DataFrame operations such as abs, map, isna, notna, replace, and others.
The TreeReduce operator is used for operations like count, sum, prod, any, all, max, min, and mean.

## Using Dynamic Partitioning in Modin

You don’t need to set any environment variables to activate this approach. Simply update to Modin 0.30.0, and your scripts for wide Dataframes will benefit from improved performance.

## Perfomance result

| operation | modin 0.29.0 | modin 0.30.0 | speed up |
| --------- | ------------ | ------------ | -------- |     
| abs       | 5.768716335  | 1.559553780  | 369.90%  |
| map       | 5.665995907  | 1.663878210  | 340.53%  |
| isna      | 4.371069111  | 1.041565318  | 419.66%  |
| notna     | 4.149922594  | 1.276469827  | 325.11%  |
| round     | 4.789841156  | 1.581816196  | 302.81%  |
| replace   | 4.871268023  | 1.442099884  | 337.79%  |
| count     | 5.163318828  | 1.835885521  | 281.24%  |
| sum       | 5.351826966  | 1.907279816  | 280.60%  |
| prod      | 5.186810397  | 2.101620920  | 246.80%  |
| any       | 5.251107819  | 1.860132668  | 282.30%  |
| all       | 5.724503774  | 1.716603592  | 333.48%  |
| max       | 5.307218991  | 1.764660481  | 300.75%  |
| min       | 5.537900437  | 1.803861558  | 307.00%  |
| mean      | 6.400612667  | 2.005258847  | 319.19%  |

To test the performance boost of Dynamic Partitioning, we generated a wide DataFrame with a shape of (20000, 5000). As you can see, the new Modin version provides significant speed-up for DataFrames with a large number of columns.

## Next Steps

Dynamic Partitioning will be implemented for other DataFrame operations as well. Stay tuned for further updates in our GitHub page and follow our posts to stay informed about Modin news.

#### Appendix

All performance measurements for this post were made on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 CPUs; 200 GB RAM).

- Modin version: 0.30.0
- Pandas version: 2.2.2
- Execution engine: Ray
- Ray version: 2.24.0
- OS: Ubuntu 22.04.2 LTS
- Python: 3.11.9
