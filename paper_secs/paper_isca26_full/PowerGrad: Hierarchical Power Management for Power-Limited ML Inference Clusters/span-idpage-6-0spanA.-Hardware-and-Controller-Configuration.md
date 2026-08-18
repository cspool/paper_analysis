# <span id="page-6-0"></span>*A. Hardware and Controller Configuration*

We evaluate the effectiveness of PowerGrad using the Cloudlab testbed [\[10\]](#page-13-24). Table [I](#page-6-2) lists our hardware configurations. We choose two types of Intel platforms: *Accelerated* is an Emerald Rapids with Advanced Matrix Instructions (AMX) [\[15\]](#page-13-25), and represents the latest architectures with accelerated support for ML; *Legacy* is a dual-CPU Haswell, and represents legacy systems still present in datacenters today to fill compute shortage. For each platform, we configure a 17 node cluster, where 16 nodes are used for ML inference and one node is the controller.

We instantiate different PowerGrad configurations on our platforms. The default two-level PowerGrad (Figure [6a](#page-6-1)) is only available for Legacy, where each node has two CPUs. This is not applicable to the Accelerated platform as it has only one CPU per node. *PG-central* (Figure [6b](#page-6-1)) is supported on both platforms, where all 16 nodes communicate with a

TABLE II ML INFERENCE APPLICATIONS.

<span id="page-6-3"></span>

| Name   | Class             | Model Type              | Low Config                  | High Config                 |
|--------|-------------------|-------------------------|-----------------------------|-----------------------------|
| Llama  | Language<br>model | Transformer             | batchsize=2<br>#tokens=40   | batchsize=8<br>#tokens=80   |
| SD     | Image<br>gener.   | Transformer<br>+ CNN    | image=128x128<br>#tokens=16 | image=512x512<br>#tokens=40 |
| VITS   | Text-to<br>speech | Transformer<br>+ 1D CNN | #tokens=40                  | #tokens=200                 |
| Resnet | Image<br>classif. | CNN                     | batchsize=4                 | batchsize=16                |

centralized controller that sets the power limits of all CPUs (16 in Accelerated and 32 in Legacy). *PG-multi* partitions the cluster into four sub-clusters of four nodes each. *PG-multi* results in a three-level hierarchy (Figure [6c](#page-6-1)) for the Legacy platform, and a two-level hierarchy (Cluster and Sub-cluster) for Accelerated.

The node, sub-cluster, and cluster controllers run every 100 ms, every 1 s, and every 4 s, respectively. We choose 100 ms for the Local Controller because RAPL measurements are not reliable if the sampling period is faster than 50 ms. To determine a stable control period for the sub-cluster and cluster controllers, we take the worst-case round-trip internode network latency (100 ms), double it, and multiply the result with the sub-cluster or cluster size. We round the result to 1 s and 4 s. We set the hyperparameters lr and α from Algorithm [1](#page-5-0) to 2.0 and 0.3, respectively.

