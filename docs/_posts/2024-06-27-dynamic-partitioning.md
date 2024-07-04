---
layout: post
title: "Dynamic Partitioning in Modin"
categories: misc
permalink: /gh_page_3.html
author: Retribution98
---

#### Annotation

In 0.30.0 version Modin introduced dynamic partitioning for DataFrames
having excessive amount of partitions into the `Map` and `TreeReduce` operators.
This allows to boost performance for certain operations by combining some partitions
into virtual ones and utilizing CPU cores more efficiently.

### Dynamic Partitioning

As you know, Modin splits a DataFrame into partitions to distribute computation.
By default, Modin uses the number or CPUs available in the system for this purpose.
When an operation is executed, the required partitions are sent to remote tasks and processed in parallel.
To achieve optimal performance it's essential to fully utilize all available CPUs.
Therefore, Modin splits the DataFrame into partitions along both axes,
sometimes resulting in an `N × N` partition grid (where N represents the number of CPUs).

This approach works well when the number of partitions close to the number of CPUs.
However, there are the cases when this method is not that effective. For straightforward operations like `map`
on a DataFrame that has the number of partitions greatly exceeding the number of CPUs,
using this partitioning strategy can lead to performance issues due to an excessively large number of remote tasks.

To address this problem, a new approach called `Dynamic Partitioning` has been implemented in Modin 0.30.0.
The main idea behind `Dynamic Partitioning` is to combine some partitions into virtual ones, thereby reducing the overall number of remote tasks.

### Using Dynamic Partitioning in Modin

You don't need to set any configuration variables to enable this approach.
Simply update to Modin 0.30.0 or higher.

### Performance result

| Operation | Modin 0.29.0(s) | Modin 0.30.0(s) | Speedup |
| --------- | ------------ | ------------ | ----- |     
| abs       | 5.768716335  | 1.559553780  | 3.70x  |
| map       | 5.665995907  | 1.663878210  | 3.41x  |
| isna      | 4.371069111  | 1.041565318  | 4.20x  |
| notna     | 4.149922594  | 1.276469827  | 3.25x  |
| round     | 4.789841156  | 1.581816196  | 3.03x  |
| replace   | 4.871268023  | 1.442099884  | 3.38x  |
| count     | 5.163318828  | 1.835885521  | 2.81x  |
| sum       | 5.351826966  | 1.907279816  | 2.81x  |
| prod      | 5.186810397  | 2.101620920  | 2.47x  |
| any       | 5.251107819  | 1.860132668  | 2.82x  |
| all       | 5.724503774  | 1.716603592  | 3.33x  |
| max       | 5.307218991  | 1.764660481  | 3.01x  |
| min       | 5.537900437  | 1.803861558  | 3.07x  |
| mean      | 6.400612667  | 2.005258847  | 3.19x  |

To test the performance boost of Dynamic Partitioning, we generated a wide DataFrame with a shape of (20000, 5000).
As you can see, the new Modin version provides significant speed-up for DataFrames with a large number of columns.

### Next Steps

There are plans to introduce `Dynamic Partitioning` into some more operators in Modin.
Stay tuned for further updates on Modin GitHub page and follow our posts to stay informed about Modin news.

#### Appendix

All performance measurements for this post were made on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 CPUs; 200 GB RAM).

- Modin version: 0.30.0
- Pandas version: 2.2.2
- Execution engine: Ray
- Ray version: 2.24.0
- OS: Ubuntu 22.04.2 LTS
- Python: 3.11.9
