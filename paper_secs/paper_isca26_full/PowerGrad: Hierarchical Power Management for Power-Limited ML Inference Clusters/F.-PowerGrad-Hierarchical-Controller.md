# *F. PowerGrad Hierarchical Controller*

A Hierarchical Controller follows the same power allocation algorithm as Algorithm [1.](#page-5-0) In this case, the algorithm takes as inputs information from each of the children nodes. A node i provides: the performance gradient G[i] aggregated using [\(13\)](#page-4-5), the average frequency of its processors f[i], the total power consumption of the node P[i], and the current power limit of the node P Lnode[i].

![](_page_6_Figure_0.jpeg)

<span id="page-6-1"></span>Fig. 6. Different organizations of PowerGrad control.

#### TABLE I EVALUATION PLATFORMS.

<span id="page-6-2"></span>

| Platform<br>Architecture |                                     | PowerGrad Configurations                                         |  |
|--------------------------|-------------------------------------|------------------------------------------------------------------|--|
| Accelerated              | Emerald Rapids (Xeon<br>Gold 5512U) | PG-central, PG-multi (without per<br>node Local Controller)      |  |
| Legacy                   | Dual-CPU<br>Haswell<br>(E5-2660 v3) | PG-central, PG-multi, PowerGrad<br>(default two-level hierarchy) |  |

PowerGrad can have any number of hierarchical levels. The default structure of PowerGrad is shown in Figure [6a](#page-6-1). It has a two-level hierarchy: Local controllers distribute the power budget across the processors in individual nodes, and a cluster controller distributes the power budget across the nodes. In addition, we also consider the Centralized (*PG-central*) and the Multi-level (*PG-multi*) variants of PowerGrad. PG-central (Figure [6b](#page-6-1)) has a single cluster-level controller that directly manages the power budgets of all the processors in the cluster. PG-multi (Figure [6c](#page-6-1)) adds an extra level in the hierarchy: it divides the cluster into multiple sub-clusters, each of which has its own sub-cluster controller.

The operations at different levels of the hierarchy are asynchronous with each other. Hence, control in the lowest level of the hierarchy is fast and is unaffected by the extra levels of the hierarchy. As we move up the hierarchy, however, communication is more expensive, as it involves using network sockets and suffering long network link latencies. As a result, higher-level controllers run less frequently.

