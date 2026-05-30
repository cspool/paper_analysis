# 4 HECATE System

#### 4.1 Architecture Overview

Figure 7 illustrates the Hecate architecture, where each device launches a runtime consisting of an executor, scheduler, dispatcher, and communicator. The executor, the main process for each device, controls the FSSDP workflow and interacts with all components. In the sharding phase, it queries the scheduler for an MoE layer sharding plan and accommodates the MoE shards containing relevant expert parameters and optimizer states on the device. In the materialization phase, it invokes the scheduler for a sparse materialization plan, launched through the communicator as sparse collectives. Additionally, when an MoE gate makes a token assignment decision in each layer, the executor queries the dispatcher for each token's destination device, as the same expert can be available on multiple devices.

The scheduler generates expert placement plans based on the expert load distribution to mitigate straggler effects. It implements two topology-aware algorithms: heterogeneous

<span id="page-6-0"></span>![](_page_6_Figure_6.jpeg)

Figure 7. HECATE's architecture.

sharding (§ 4.3) and sparse materializing (§ 4.2) for the two phases of FSSDP, respectively. The dispatcher determines each token's destination device based on token assignment of MoE gates and the current materialized MoE layer parameter placement (§ 4.4). The communicator handles assigned communication tasks by maintaining a queue and scheduling them to the runtime communication library (e.g., NCCL [5]), executing the dispatching plan as an All-to-All collective.

#### <span id="page-6-1"></span>4.2 Sparse materialization

In the materialization phase of each MoE layer, FSSDP requires a sparse materialization plan  $\mathcal{P}'$  as the target placement for spAG( $\mathcal{P},\mathcal{P}'$ ) to perform sparse communication of MoE layer parameters. Here, the chunk placement is  $\mathcal{P} = \mathcal{E} \times \mathcal{D}$ , where  $\mathcal{E} = \{E_1, E_2, \ldots\}$  represents the parameters for all experts in an MoE layer as the collective logical buffer, with each expert serving as a chunk.

Algorithm 1 presents the topology-aware sparse materialization algorithm, used by HECATE's scheduler to heuristically search for a near-optimal parameter placement under two system constraints: overlap degree t and memory capacity m. The overlap degree t represents the maximum number of experts that can be materialized on other devices with the communication overhead completely hidden in attention layers. According to Equation 1, it can be calculated by  $t = T_{\text{non-MoE}} \cdot \frac{\text{bw}}{\text{expert\_size}}$ , where  $T_{\text{non-MoE}}$  is computation latency of previous non-MoE layers (e.g., the attention layer) and expert\_size is an expert's parameter byte size. Critically, bw reflects the cluster's interconnect topology. When the cluster features heterogeneous interconnects with significant bandwidth differences between inter-node and intra-node communication, by represents the inter-node bandwidth, as the algorithm prioritizes minimizing crossnode communication. If the interconnect is homogeneous, bw reflects the uniform inter-device bandwidth. The memory capacity denotes the maximum number of experts that can be materialized on each device's available memory. These two integers are profiled by the scheduler and passed to the algorithm as input. Expert load *F* is estimated using a sliding window average over the latest w iterations (Hecate uses w = 5).

The two outermost branches in Algorithm 1 represent different conditions of system constraints. When the overlap degree is less than or equal to the memory capacity (lines 4 to 5), the algorithm materializes as many overloaded experts as possible on all devices within the overlappable time. Otherwise (lines 6 to 11), the algorithm sparsely materializes experts on devices according to their load distribution. Experts with higher loads are materialized on more devices (line 9), prioritizing nodes that do not already have the expert parameters materialized (line 10). This topology-aware design, which considers the potential bandwidth disparities between inter-node and intra-node links, helps mitigate

All-to-All straggler effects due to inter-node communication congestion.

The sparse materialization can include a *calibration* stage additionally, occurring immediately after the MoE gate generates the token assignment decision. Since the overlapped sparse materialization is based on an estimated expert load distribution, the current distribution (i.e. the real-time token assignment decision) can still vary due to the stochastic nature of training. The *calibration* re-runs Algorithm 1 with the latest expert loads and remaining memory capacity to determine if an additional SparseAllGather can be executed to further reduce load imbalance. If the calibrated placement results in a lower latency, considering the additional communication overhead on the training critical path, the scheduler will accept and return the placement plan for the communicator to execute before initiating token dispatching.

#### **Algorithm 1:** Sparse Materialization

```
Input: \mathcal{P}: sharded parameter placement
                F: expert load distribution
                t: overlap degree
                m: memory capacity per device
    Output: \mathcal{P}': materialization plan
1 t \leftarrow \min(t, |\mathcal{E}|), m \leftarrow \min(m, t);
_{2}\mathcal{P}^{\prime}\leftarrow\mathcal{P};
3 if t \leq m
          \mathcal{E}^{\text{topT}} \leftarrow \text{Top } t \text{ experts by load } F;
          \mathcal{P}' \leftarrow \mathcal{P}' \cup (\mathcal{D} \times \mathcal{E}^{\text{topT}}):
5
6 else
          totSlots \leftarrow |\mathcal{D}| \cdot m;
          foreach e \in \text{sortByLoadDescending}(\mathcal{E}^{topT}) do
                n \leftarrow assignSlotsByLoad(e, totSlots, F);
                \mathcal{P}^e \leftarrow \text{Distribute } n \text{ replicas of expert } e \text{ across}
10
                  nodes and devices, prioritizing nodes with
                  more available slots;
                \mathcal{P}' \leftarrow \mathcal{P}' \cup \mathcal{P}^e:
11
12 return \mathcal{P}'
```

#### <span id="page-7-7"></span><span id="page-7-6"></span><span id="page-7-5"></span><span id="page-7-0"></span>4.3 Heterogenous Sharding

The design of Hecate's sparse materialization primarily benefits the overloaded experts, as their placements are more likely to be materialized on multiple devices. However, the placement of underloaded experts can also be optimized to further reduce straggler effects, particularly when training with multiple nodes. For instance, if a node contains MoE shards with only underloaded experts, the inbound bandwidth of this node may be oversubscribed by All-to-All for these crowded underloaded experts to receive their tokens, as the node is likely the sole destination for these tokens.

Heterogeneous sharding algorithm is introduced in Hecate for sharding MoE layers across devices in the sharding phase,

<span id="page-7-8"></span>![](_page_7_Figure_7.jpeg)

Figure 8. Homogeneous vs. Heterogeneous Sharding

determining better placements for the underloaded experts. The algorithm is *heterogeneous* since it allows an MoE shard to have an arbitrary number of experts (ranging from 0 to  $|\mathcal{E}|$ ) while maintaining memory balance across devices, as depicted in Figure 8. In Hecate, MoE layers are initialized using homogeneous sharding (i.e. even sharding), and periodically re-sharded in heterogenous manners during training.

Algorithm 2 presents the sharding algorithm. It schedules all MoE layers in the PTM collectively to ensure even memory demand for sharding all layers across devices. As the algorithm involves cross-layer scheduling, some variables in the algorithm pseudocode may have the superscript "g" to indicate that they cover all MoE layers (e.g.  $\mathcal{E}^g$ ), while the index "l" is used to denote variables specific to a particular MoE layer l (e.g.  $\mathcal{E}_l$ ). It returns the sharding plan for all MoE layers in the form of  $\mathcal{P}^g = \{\mathcal{P}_0, \mathcal{P}_1, \cdots, \mathcal{P}_L\}$ , where each element is an expert placement for the parameters and optimizer states of the corresponding MoE layer.

Experts first partitioned into two disjoint sets layer-wisely (from line 1 to line 2):  $\mathcal{J}$  are overloaded experts that can be selected by the sparse materialization, and  $\mathcal{J}'$  contains the remaining experts that are not "overlappable". The algorithm initializes same number of slots per device (line 3) for plugging in experts while ensuring consistent memory demand across devices. Experts in  $\mathcal{J}'$  are scheduled firstly, layer by layer (from line 6 to line 14), prioritizing layers with the most overloaded expert. For each expert, the algorithm first attempts to find the least-loaded node. If multiple nodes have the same lowest load, the node with fewer available slots is prioritized. The algorithm then tries to find the least-loaded device on the selected node, using the same priority rule. The expert is then assigned to the device, and the available slots on the device are decreased. Finally, the algorithm fills the remaining slots with experts from  $\mathcal{J}$  (line 16).

It is important to note that unlike sparse materialization, the heterogeneous sharding of Hecate introduces resharding latency to the training critical path, which may

seem to result in the timeliness challenge. However, we argue that re-sharding can be performed at a low frequency, amortizing the overhead over iterations. Since it focuses on the placement of underloaded experts, which are trained with fewer tokens per iteration, the MoE gate will have gradients with smaller magnitudes corresponding to these experts. This implies that the loads of underloaded experts change slowly (confirmed by Figure 3). Consequently, re-sharding can be triggered less frequently, extracting the last bit of performance improvement from the FSSDP sharding design.

