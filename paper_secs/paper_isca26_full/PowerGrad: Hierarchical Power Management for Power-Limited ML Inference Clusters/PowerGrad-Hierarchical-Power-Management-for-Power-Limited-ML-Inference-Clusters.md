# PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters

Hyoungwook Nam, Raghavendra Pradyumna Pothukuchi\*, Alper Buyuktosunoglu<sup>†</sup>, Aporva Amarnath<sup>‡</sup>, Pradip Bose<sup>†</sup>, Josep Torrellas

{hn5, torrella}@illinois.edu, raghav.pothukuchi@unc.edu, {alperb, pbose}@us.ibm.com, aporva.amarnath@amd.com University of Illinois Urbana-Champaign \*University of North Carolina Chapel Hill †IBM Research ‡AMD Research

Abstract—As machine learning (ML) workloads demand more power and datacenters integrate renewable energy, workloads have to deal with situations where power demands exceed supply. In such situations, intelligently allocating the power among the nodes is key to maximizing efficiency. However, this is hard to do for ML inference workloads, where system administrators cannot profile the workload ahead of time.

To address this challenge, this paper proposes *PowerGrad*, a hierarchical power-management framework for power-limited ML inference clusters. The idea is to dynamically identify the *performance gradient* of each running workload, which characterizes the performance sensitivity of the workload to power changes. At runtime, a Gradient Estimator collects hardware measurements and uses them to estimate performance gradients. Then, to maximize efficiency, Local Controllers and Hierarchical Controllers re-distribute the power from low-gradient workloads to high-gradient ones within a node and across nodes, respectively. PowerGrad is especially effective for *severely power-limited environments*, where every node demands more power than its maximum allocation.

While PowerGrad can be applied to a variety of compute architectures, it needs dynamic hardware performance counter information that is unavailable in GPUs and accelerators. Consequently, we demonstrate PowerGrad on two CPU clusters running popular ML inference workloads in power-limited setups. The results show that PowerGrad is both effective and easily retargetable across different architectures. In traditional dual-CPU nodes, PowerGrad reduces the average and tail latencies by a mean of 22.9% and 23.0%, respectively, relative to the strongest of a set of software-transparent baselines. In single-CPU nodes with ML acceleration support, PowerGrad reduces the average and tail latencies by a mean of 9.0% and 9.9%, respectively.

Index Terms—Power management, Machine learning, Distributed systems, Hierarchical design

#### I. INTRODUCTION

The increasing computing needs of machine learning (ML) workloads have fueled a rapid expansion of datacenters and led to a dramatic rise in energy consumption [4], [16], [36]. As a result, in an effort to enhance sustainability, datacenters have started to use renewable energy sources [6], [25]. In such environments, there is a higher chance of situations where the energy available to a cluster is less than the demand, due to fluctuations [32] or due to Demand-Response (DR) actions [45]. In such power-limited environments, the allocation of power among the nodes of the cluster plays a critical role in overall efficiency.

This issue compounds the already existing challenge of effective power management in ML inference environments.

Such environments are typically dynamic and unpredictable, as a datacenter usually serves a heterogeneous set of ML models concurrently, and the mix of models changes frequently. Moreover, for a given model, users launch a variety of requests that have different power and performance characteristics based on their inputs [38]. As a result, a system administrator cannot profile the workload ahead of time. To complicate matters, many requests execute for only a few seconds [26], [29]. Overall, power management in ML inference clusters needs to be dynamic, scalable, agile, and not dependent on workload-specific parameters.

Unfortunately, most existing methods for datacenter power control do not satisfy these requirements. First, many of them are not software-transparent, in that they either rely on application profiling (e.g., [24], [33], [43], [44]) or work only for very specific types of applications [46]. We cannot apply such algorithms to ML inference clusters because it is impossible to profile all combinations of ML models and user requests ahead of time.

Second, while there are software-transparent methods that rely on heuristics based on power usage (e.g., [8], [35], [37]), these heuristics are not intelligent enough to maximize power efficiency in ML clusters. They monitor power use but do not identify which workloads are compute-bound and which are memory-bound—which is minimally needed to assign the power in a more efficient manner. Methods that simply rely on the power consumed are especially ineffective in *severely power-limited environments*, where every node demands more power than its maximum allocation. In this case, it becomes very hard to determine the best priority assignment between the nodes using only the power consumption patterns.

Finally, many methods use centralized algorithms that require information from all the nodes at every time step. These algorithms are typically too slow in large-scale datacenters with rapidly-changing workloads. Hence, they are not scalable.

To address these challenges, this paper proposes an intelligent and scalable power management framework for power-limited ML inference clusters. We call it *PowerGrad*. Instead of profiling the workloads ahead of time, PowerGrad reads the hardware performance counters of processors at runtime. With these values, a *Gradient Estimator* dynamically estimates the *performance gradient* of each running workload. This is the performance sensitivity of the workload to power changes at a given time. Then, to maximize efficiency, power controllers

re-distribute the power from power-insensitive workloads to power-sensitive ones. In other words, they move power from the workloads that lose less performance per unit of power loss to the workloads that gain more performance per unit of power gain. The result is a net performance gain for the same total power. This is a novel approach that enables setting priorities among the workloads when all workloads demand more power than their allocations. Hence, PowerGrad is especially effective at severely power-limited environments.

In PowerGrad, controllers are organized hierarchically. *Local Controllers* re-distribute the power between processors in the same node. They also pass the local gradients and measurements to *Hierarchical Controllers*, which re-distribute the power across the nodes in the cluster or sub-cluster. This approach enables finer time granularity of actuation at the local levels, which is crucial to handle rapidly-changing ML workloads. Local and hierarchical controllers use the same gradient-based algorithm, so that it is easy to recursively add extra levels of hierarchy to a system.

PowerGrad's framework is easily retargetable to a variety of computer architectures. For a new architecture, one only needs to re-train the Gradient Estimator. However, PowerGrad needs to dynamically collect hardware performance counter information during application execution, to adjust the power allocation at runtime. This support is not available at runtime during kernel execution in current GPUs and accelerators. For example, NVIDIA's profiling tool allows the querying of performance counters only after the kernel execution completes [\[27\]](#page-13-15). Hence, we can only evaluate PowerGrad on CPUs.

To show PowerGrad's portability, we evaluate it on CPU clusters of two types of CPU architectures: traditional CPUs without ML acceleration support (Intel Haswell) and newer CPUs with ML acceleration (Intel Emerald Rapids). The clusters run a set of ML inference workloads.

Our results show that PowerGrad is very effective for powerlimited environments. Thanks to PowerGrad's hierarchical nature, it is more effective when there are multiple processors in each node. On a 16-node cluster of dual-CPU Haswell nodes with a range of power budgets, PowerGrad reduces the average and tail latencies by a mean of 22.9% and 23.0%, respectively, relative to the strongest of a set of state-of-theart software-transparent power-management baselines. On a 16-node cluster of single-CPU Emerald Rapids nodes with ML acceleration and a range of power budgets, PowerGrad reduces the average and tail latencies by a mean of 9.0% and 9.9%, respectively. In severely power-limited environments, the relative gains of PowerGrad increase: for 55W per Haswell node, the latency reductions are 23.6% and 27.4%, while for 115W per Emerald Rapids node, the reductions are 18.3% and 20.2%.

The contributions of this work are as follows:

- The PowerGrad hierarchical power-management framework for power-limited ML inference clusters.
- A mathematical method to estimate performance gradients from runtime performance hardware measurements.

![](_page_1_Figure_8.jpeg)

<span id="page-1-0"></span>Fig. 1. Power-performance patterns of four ML inference applications.

- A control algorithm to shift power from power-insensitive workloads to power-sensitive ones in a hierarchical manner.
- Implementation and evaluation of PowerGrad in CPU clusters of different architectures.

