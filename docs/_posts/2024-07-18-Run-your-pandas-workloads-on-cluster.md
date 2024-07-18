---
layout: post
title: "Running Pandas on multi node cluster with Modin"
categories: misc
permalink: /gh_page_4.html
author: arunjose696
---

## Introduction

Handling large datasets efficiently is a common challenge in data science. Traditional tools like Pandas can struggle with scalability, often becoming slow and memory-intensive with large datasets. This is where Modin and Ray come into play. Modin is a scalable dataframe library that accelerates Pandas operations by distributing the workload across multiple cores or nodes. Ray, a distributed computing framework, further enhances this scalability.

### Modin
Modin is designed to scale Pandas seamlessly, allowing you to utilize all available hardware resources with minimal code changes. Simply replace your Pandas import with Modin, and you're ready to go.

### Ray
Ray is the default backend for Modin. When operating in local mode (without a cluster), Modin will create and manage a local cluster using Dask or Ray. A key feature of Ray is its capability to extend a Ray cluster across multiple nodes. This means you can efficiently process your data workloads using a cluster of smaller machines, rather than relying on a single large machine.
In contrast, Pandas processes data using a single thread and a single node. For memory-intensive tasks that cannot be handled by a single node, Pandas will encounter a memory error. To demonstrate this, I used the NY taxi dataset, which I synthetically expanded by replicating it 10 times to generate a large volume of data.


### Work load
I would be using the below script involving three operations: read_csv, apply, and map for my experiments. I have chosen operations that do not require data transfer between nodes to minimize data transfer across nodes. As the main goal here is to run the workload on cluster.

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

### Running script on pandas 

To run this workload, I chose a t3.2xlarge instance, which has 32GB of RAM and can hold the whole data in memory. However when processing the script, Pandas encounters an Out of Memory (OOM) error on the t3.2xlarge instance and crashed because the processing dataframe required more memory than a single node could provide. The original data fit into memory, but the additional memory needed during processing caused the issue. Since a single node was insufficient for the workload, the options were to scale the system either horizontally (using multiple nodes) or vertically (using a more powerful machine). Because Pandas operates only on a single node, horizontal scaling isn't an option. However, Modin on Ray can achieve this by distributing the workload across multiple nodes. 

### Prerequisites

The seamlessness of Modin allows you to execute the same code as your Pandas code on a cluster. You can easily submit the python script as a Ray job to a Ray cluster. However, prerequisites include:
1.	A Ray cluster must be set up.
2.	The dataset must be available on all nodes in the cluster.

#### Setting Up Ray Cluster and Distributing the Dataset
There are multiple ways to spawn a Ray cluster. You can have your on-premise Ray cluster, or an easy alternative is to set up a cluster in AWS using tools such as [KubeRay](https://github.com/ray-project/kuberay/tree/master/helm-chart/ray-cluster). In my configuration, I used [init containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) that download and prepare data on all nodes of the Ray cluster, satisfying both prerequisites.

Once the Ray cluster is up and running, set the RAY_ADDRESS environment variable to the Ray dashboard URL on your client machine.
```bash
export RAY_ADDRESS=URL_OF_RAY_DASHBOARD
```
Now, make minor changes to your Python script by importing Pandas from Modin and initializing Ray.

```python
import modin.pandas as pd
import ray
ray.init(address="auto", logging_level="WARNING")
# your workload without any changes
```



You can view the complete Python script below.
<details>
  <summary>modin_workload.py</summary>

    <pre><code class="language-python">
    # import pandas as pd
    import modin.pandas as pd
    import ray
    from modin.utils import execute # execute is just used to make sure all the asynchronous operations are complete as we benchmark.
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


### Results and discussion

I observed that by scaling up the number of cluster nodes, Modin could execute the workload with 4 nodes and above. It was also noted that Modin shows performance improvements when scaling the number of nodes from 4 to 32.
As Modin works with 4 nodes and both Pandas and Modin do not work with fewer than 3 nodes, a possible conclusion is that the workload requires almost 4 times the memory(32Gb) of a single t3.2xlarge instance. To provide an apples-to-apples comparison for the performance gain that can be obtained by Modin compared to Pandas, we decided to benchmark the script on an instance with more than 120 GB of memory. We selected an r6a.4xlarge instance, which comes with 128GiB RAM.

With Pandas, this script takes 487.444 seconds to execute. The graph below shows the time taken by Modin to execute the script when run with different cluster sizes. The x-axis shows the number of nodes in the Ray cluster. Each of the trend lines shows the number of CPUs to be used by Ray in each node. Modin, by default, sets the number of workers equal to the total number of CPUs in the Ray cluster.



### Performance on a Single Node
From the graph, it appears that Modin on a single node is initially slower than Pandas if we use 4 or fewer workers (leftmost points of orange, grey, and yellow lines). This could be because we have very few Ray workers to provide sufficient parallelism for speed up compared to Pandas. Even with a single node, if we utilize all of the CPUs (or even 8 CPUs), we can see Modin offers a performance improvement compared to Pandas.

### Performance on Scaling Nodes to 32

We can see the performance continues to increase if we add more nodes to the cluster. Using a 32-node cluster with all 16 CPUs utilized in all nodes executes the script in a significantly reduced time, which is a significant performance improvement compared to executing the workload on a single node.
<div class="centered-image">
    <img src="imgs/blog_post_4/Modin_multiple_nodes.png" alt="Perf Results multinode" >
</div>

#### Appendix

All performance measurements for this post were made on Intel(R) Xeon(R) Gold 6238R CPU @ 2.20GHz (112 CPUs; 200 GB RAM).

- Modin version: 0.30.0
- Pandas version: 2.2.2
- Execution engine: Ray
- Ray version: 2.9.2
- OS: Ubuntu 22.04.2 LTS
- Python: 3.9
