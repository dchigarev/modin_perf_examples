---
layout: post
title: "Scale-out pandas beyond a single node with Modin"
categories: misc
permalink: /gh_page_4.html
author: arunjose696
---

## Annotation

Handling large datasets efficiently is a common challenge in data science.
Traditional tools like pandas can struggle with scalability, often becoming slow and
memory-intensive with large datasets. This is where Modin on Ray comes into play.
Modin is a powerful dataframe library designed to scale pandas operations by spreading the workload
across multiple cores or nodes. It supports various execution engines such as Ray, Dask and MPI,
with Ray being the default choice. Ray a distributed computing framework ensures smooth and
efficient data processing.

### Modin

Modin is designed to scale pandas seamlessly, allowing you to utilize all available hardware resources with minimal code changes. Simply replace your pandas import with Modin, and you're ready to go.

```python
# import pandas as pd
import modin.pandas as pd
```

### Ray

Ray is the default execution engine in Modin, and it shines in handling data across multiple nodes.
Even when running locally in a single node, Modin sets up a local Ray cluster to manage the workload.
One of Ray's standout features is its ability to extend across multiple nodes, allowing you to process large
data workloads using a cluster of smaller machines instead of relying on a single, massive machine.

In contrast, pandas operates on a single thread and node, which means that for memory-intensive tasks,
it can quickly run into memory errors if a single node isn't sufficient. To illustrate this, we used the NY Taxi dataset,
expanding it synthetically by replicating it 10 times to create a dataset large enough to push the limits of a single-node setup.

### Workload

For the experiments, we will be using a script that involves three operations: `read_csv`, `apply`, and `map`.
These operations have been selected to minimize data transfer between nodes, ensuring that the primary
focus is on executing the workload efficiently within a cluster. The aim is to optimize performance by
leveraging cluster capabilities without the added complexity of data movement between nodes.

<details>
  <summary>pandas_workload.py</summary>

  <pre><code class="language-python">
    import pandas as pd

    file_path = "/home/ray/data/big_yellow.csv"
    df = pd.read_csv(file_path)
    df = df.map(str)
    result = pd.DataFrame()

    payment_type_map = {
        1: 'Credit Card',
        2: 'Cash',
        3: 'No Charge',
        4: 'Dispute',
        5: 'Unknown',
        6: 'Voided Trip'
    }

    result["df_payment_type"] = df['payment_type'].map(payment_type_map)

    def custom_function(row):
        return str(row["passenger_count"]) + " passengers were picked up at " + str(row["tpep_pickup_datetime"])

    result["description"] = df.apply(custom_function, axis=1)

  </code></pre>
</details>

### Running script with pandas 

To run this workload, we chose a t3.2xlarge instance with 32GB of RAM, enough to hold all the data in memory.
However, when processing the script, pandas encountered an Out of Memory (OOM) error and crashed
because the processing dataframe required more memory than a single node could provide.
The original data fits into memory, but the additional memory needed during processing caused the issue.
Since a single node was insufficient for the workload, we had two options: scale horizontally
(using multiple nodes) or vertically (using a more powerful machine). pandas, limited to a single node,
can't scale horizontally. Fortunately, Modin on Ray can distribute the workload across multiple nodes,
making horizontal scaling possible.

### Prerequisites

Modin makes it a breeze to run your existing pandas code in a cluster with minimal changes.
You can simply submit your Python script as a Ray job to a Ray cluster. Just keep in mind a couple of prerequisites:

1.  You need to have a Ray cluster up and running.
2.  The dataset must be available (not just accessible) on all nodes in the cluster.

#### Setting Up Ray Cluster and Distributing the Dataset

There are several ways to set up a Ray cluster. You can opt for an on-premise setup or
take the easier route of configuring a cluster in AWS using tools like
[KubeRay](https://github.com/ray-project/kuberay/tree/master/helm-chart/ray-cluster).
In our configuration, we used [init containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/)
to download and prepare data on all nodes of the Ray cluster, ensuring that both prerequisites are effectively met.

Once the Ray cluster is up and running, set the  `RAY_ADDRESS` environment variable
to the Ray dashboard URL on your client machine.

```bash
export RAY_ADDRESS=URL_OF_RAY_DASHBOARD
```

Now, make minor changes to your Python script by importing pandas from Modin and initializing Ray.

```python
import modin.pandas as pd
import ray

ray.init(address="auto", logging_level="WARNING")
```

You can view the complete Python script below.

<details>
  <summary>modin_workload.py</summary>

    <pre><code class="language-python">
    # import pandas as pd
    import modin.pandas as pd
    from modin.utils import execute # execute is just used to make sure all the asynchronous operations are complete as we benchmark.
        import ray

    ray.init(address="auto", logging_level="WARNING")
    file_path = "/home/ray/data/big_yellow.csv"
    df = pd.read_csv(file_path)
    df = df.map(str)
    result = pd.DataFrame()

    payment_type_map = {
        1: 'Credit Card',
        2: 'Cash',
        3: 'No Charge',
        4: 'Dispute',
        5: 'Unknown',
        6: 'Voided Trip'
    }

    result["df_payment_type"] = df['payment_type'].map(payment_type_map)

    def custom_function(row):
        return str(row["passenger_count"]) + " passengers were picked up at " + str(row["tpep_pickup_datetime"])

    result["description"] = df.apply(custom_function, axis=1)
    execute(result)
    </code></pre>

  ```
</details>

### Results

We can observe that by increasing the number of nodes in the cluster, Modin could handle the workload
effectively with 4 nodes or more. Significant performance improvements were evident as we scaled
from 4 to 32 nodes. Given that Modin operates efficiently with 4 nodes and both pandas and Modin
struggle with fewer than 3 nodes, it suggests the workload demands nearly 4 times the memory (32GB) of
a single t3.2xlarge instance.

To provide a fair comparison of the performance gains offered by Modin over pandas,
we decided to benchmark the script on an instance with over 120 GB of memory.
We chose an r6a.4xlarge instance, which boasts 128GiB of RAM.

With pandas, the script took almost 400 seconds to execute, which serves as our baseline.

### Performance on a Single Node

The graph reveals an interesting twist: Modin on a single node starts off slower than pandas
when using 4 workers or fewer. This could be because the overhead of distributing data and spinning up
the Ray cluster doesn't pay off with so few workers. It's like trying to split chores among two people
when you really need a full team. But here's where it gets exciting: crank up the CPU count
(even just to 8), and Modin shifts into high gear, outperforming pandas even on a single node.
Talk about a comeback!

<img src="imgs/blog_post_4/Modin_single_node.png" alt="Perf Results single node"  style="display: block; margin-left: auto; margin-right: auto;">


### Performance on Scaling Nodes to 32

The performance continues to soar as we add more nodes to the cluster. With a 32-node cluster,
utilizing all 16 CPUs in each node, the script executes in a fraction of the time it takes on a single node.
This results in a dramatic performance boost, showcasing the impressive scalability of Modin.

<img  src="imgs/blog_post_4/Modin_multiple_nodes.png" alt="Perf Results multinode"  style="display: block; margin-left: auto; margin-right: auto;">

#### Appendix

All performance measurements for this post were made on an AWS r6a.4xlarge instance.

- Modin version: 0.30.0
- Pandas version: 2.2.2
- Execution engine: Ray
- Ray version: 2.9.2
- OS: Ubuntu 20.04.6 LTS
- Python: 3.9.18
