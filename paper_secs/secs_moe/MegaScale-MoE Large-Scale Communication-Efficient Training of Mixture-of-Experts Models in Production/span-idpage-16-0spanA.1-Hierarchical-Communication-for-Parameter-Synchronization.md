# <span id="page-16-0"></span>A.1 Hierarchical Communication for Parameter Synchronization

Let the full attention weights size be P, the dimension of model parallelism (TP or SP) be n, and the data parallel size be d. Typically, GPUs for model parallelism are located on the same node, requiring intra-node communication, whereas data parallelism spans across nodes, requiring inter-node communication. Consider a data parallelism group containing d devices, each holding the identical partition of the parameter.

For parameter synchronization in TP attention, communication involves data of size P/n across d devices in two primary steps in LLM training:

- inter-node reduce-scatter operation, where the data size is P/n, on d devices.
- inter-node all-gather operation, where the data size is P/n, on d devices.

leading to primarily inter-node communication, with a communication volume of 2P/n(d-1)/d.

With SP attention, the parameter synchronization involves the entire data of size P across  $n \times d$  devices. Considering the discrepancy between intra-node and inter-node network bandwidth, this process can be implemented by four-step hierarchical communication, where the replicated parameters are first reduced within a node and then reduced across nodes, before being distributed back to each device. Figure 5a illustrates a hierarchical communication example where n=3 and d=2. The detailed steps are as follows.

- intra-node reduce-scatter operation, where the data size is *P*. on *n* devices.
- inter-node reduce-scatter operation, where the data size is P/n, on d devices.
- inter-node all-gather operation, where the data size is *P/n*, on *d* devices.
- intra-node all-gather operation, where the data size is *P*, on *n* devices.

The inter-node communication volume in SP attention remains at 2P/n(d-1)/d, with additional intra-node volume of 2P(n-1)/n.

Moreover, due to the distinct resources for intra-node and inter-node communications, these steps can be segmented into small chunks and pipelined to efficiently hide each other as shown in Figure 5b. The ratio of inter-node communication latency and intra-node communication latency is

$$\frac{1}{n} \times \frac{\text{intra-node bandwidth}}{\text{inter-node bandwidth}} \times \frac{n(d-1)}{d(n-1)}$$
 (10)

Consider a typical training scenario involving an H100 SXM machine, where the NVLink bandwidth is 450 GB/s, and the inter-device NIC communication bandwidth is 50 GB/s. In this context, the latency of inter-node communication can easily surpass that of intra-node communication. This implies that the communication within a node can overshadow that between nodes. Consequently, in such scenarios, the synchronization of gradients and parameters with SP attention is, in fact, consistent with TP attention.