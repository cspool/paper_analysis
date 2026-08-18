# B. Evaluation Methodology

We implement a custom, topology-aware analytical simulator tailored to NAHP and wafer-scale fabrics. As illustrated

in Fig. 11, the simulator takes as input (i) the brain model parameters (regions, neurons, and connectivity), (ii) the evaluated processing paradigm (data structure, routing scheme with packet format), and (iii) the modeled hardware parameters (interconnect topology and router/link timing and throughput). It constructs the NoC and derives the per-node workload, then performs storage analysis and routing-based traffic simulation. Moreover, it aggregates expected spike events over a 1 ms step and evaluates per-step communication latency with topologydependent propagation delay. To formalize these computations, we model the neuromorphic system as a graph G = (V, E), where each vertex  $v_i \in V$  represents a neuromorphic processing node and each edge  $e\langle v_i, v_i \rangle \in E$  denotes a physical communication link between two nodes. For each neuron  $n_i$ , we denote its residing node as  $v(n_i)$ , and let  $V_d(n_i)$  be the set of destination nodes where its post-synaptic neurons reside. The evaluation metrics are derived in three parts: communication load/traffic, storage footprint, and per-step latency.

a) Communication.: The routing cost of a firing neuron  $n_i$  depends on the routing scheme. For broadcast, the cost is destination-independent and defined as

$$L_b(n_i, V_d(n_i)) = |V_b| - 1, (1)$$

where  $V_b$  denotes the set of nodes within the broadcast domain. In neuron-centric,  $V_b$  includes all nodes V, while NAHP is restricted to nodes within the local region. For unicast, the cost is determined by the hop distance between the source node and each destination:

$$L_u(n_i, V_d(n_i)) = \sum_{v_j \in V_d(n_i)} \text{Dist}(v(n_i), v_j),$$
 (2)

where  $\mathrm{Dist}(v(n_i),v_j)$  indicates the hop distance between the injecting node  $v(n_i)$  and the destination node  $v_j$ . Since unicast transmissions are independent point-to-point deliveries,  $L_u$  equals the total sum of hop distances from  $v(n_i)$  to all  $v_j \in V_d(n_i)$ . In axon-centric processing,  $v(n_i)$  is the source node that injects the unicast packets, whereas in NAHP the unicast packets are injected by the boundary nodes assigned to the source neuron.

The average routing load R denotes the number of packets processed per router per second. We categorize synapses into local (intra-region) and global (inter-region) connections, denoted by destination sets  $V_d^L(n_i)$  and  $V_d^G(n_i)$ , respectively. The average per-node load is given by

$$R = R_{\text{Local}} + R_{\text{Global}},\tag{3}$$

$$R_{\text{Local}} = \frac{\lambda}{|V|} \sum_{n_i} L(n_i, V_d^L(n_i)), \tag{4}$$

$$R_{\text{Global}} = \frac{\lambda}{|V|} \sum_{n_i} L(n_i, V_d^G(n_i)), \tag{5}$$

where  $R_{\rm Local}$  and  $R_{\rm Global}$  denote the average packet rate per node due to local and global synaptic transmissions, with  $\lambda$  representing the firing rate. Since both neuron-centric and NAHP employ broadcast for local connections,  $R_{\rm Local}$  remains

TABLE III

COMPARISON OF NEUROMORPHIC PROCESSING PARADIGMS

| Paradigm                  | Neuron-Centric                         | Axon-Centric                                                            | Neuron-Axon Hybrid (Ours)                           |
|---------------------------|----------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------|
| Routing Method            | Broadcast (or Multicast <sup>1</sup> ) | Unicast                                                                 | Local Broadcast + Global Unicast                    |
| Routing Path Reuse        | Yes                                    | No                                                                      | Yes                                                 |
| Destination Scope         | All chips                              | Only destination                                                        | Local Region + Destination                          |
| Event Representation      | FNid (37bit)                           | NodeID&FAid (17+29bit)                                                  | LNid (27bit) + NodeID&GAid (17+23bit)               |
| Synapse Storage           | Adjacency List                         | Adjacency List (or Crossbar*)                                           | Local List + Global List                            |
| Synapse Addressing Scheme | FNid (237 entries)                     | FAid (2 <sup>27</sup> entries)                                          | LNid ( $2^{27}$ entries) + GAid ( $2^{23}$ entries) |
| Addressing Entry Width    | 37bit                                  | 37bit                                                                   | Local(37bit) + Global(33bit)                        |
| Synapse Addressing Cost   | High                                   | Low                                                                     | Low                                                 |
| Typical Processors        | SpiNNaker [29], Darwin [26]            | Loihi [5], TrueNorth* [1],<br>Tianjic* [6], Darwin3 [25], PAICORE* [43] | This work                                           |

<sup>[1]</sup> Multicast requires large routing tables for each node, which is not scalable to billion-scale systems.

identical. For global connections, our boundary-triggering mechanism reduces redundant hops, yielding lower  $R_{\rm Global}$  compared to axon-centric systems. The average traffic (in bits) is then computed as:

$$T = R_{\text{Local}} \times W_L + R_{\text{Global}} \times W_G, \tag{6}$$

where  $W_L$  and  $W_G$  denote the packet sizes for local and global events (Table III). Neuron-centric systems use FNids for both  $W_L$  and  $W_G$ , while axon-centric systems require NodeID+FAids. Our hybrid scheme employs LNids for local broadcast, reducing  $W_L$  by 10 bits in the 100B model, and GAids for global unicast, reducing  $W_G$  by 4 bits.

*b)* Storage.: The storage can be expressed as the sum of neuron states, synaptic weights, and indexing structures:

$$S_{\text{total}} = S_{\text{neuron}} + S_{\text{synapse}} + S_{\text{index}},$$
 (7)

Let N denote the number of neurons per node, F the average fan-out,  $B_n$  the storage per neuron (bits), and  $B_s$  the storage per synapse (bits). Then

$$S_{\text{neuron}} = N \cdot B_n, \ S_{\text{synapse}} = N \cdot F \cdot B_s,$$
 (8)

In our experiments,  $N=1.3-1.49\times 10^6$  (requiring 21 bits), F=256, and  $B_n=64$  bits. The synaptic weight is represented with 16 bits. Each synapse stores the target neuron ID together with its weight, giving  $B_s=21+16=37$  bits. Accordingly,  $S_{\rm neuron}=10.4$  MB and  $S_{\rm synapse}=1.54$  GB, showing that synaptic weights dominate overall storage.

The indexing overhead  $S_{\text{index}}$  varies with the processing paradigm, since the primary cost comes from synapse addressing entries. In neuron-centric and axon-centric methods, adjacency lists are maintained for both local and global synapses. As summarized in Table III, both requires each entry to store a 37 bit synapse addressing entry (29 bit address + 8 bit fanout). In contrast, the NAHP separates local and global synapses: local entries are proportional to the number of neurons within the region, while global entries, only about 5% of the total, are indexed separately with 4 fewer address bits. In terms of entry count, the neuron-centric paradigm stores the full neuron entries at every node, leading to an overhead that scales with the model size. The axon-centric paradigm

requires storing only the entries corresponding to the total fanin connections. The NAHP paradigm partitions addressing into local and global domains: local entries are equal to the number of neurons within a region, while the remaining global axon entries are maintained separately.

c) Latency.: Cycle-accurate latency simulation is intractable at whole-brain scale. We therefore use a step-level latency model that abstracts away fine-grained queuing dynamics within a 1 ms step, and defines the communication latency of a single simulation step (excluding computation) as the sum of the service time at the most heavily loaded router and the propagation delay along the longest path:

$$Latency = \frac{T_{\text{max}}}{\Theta_{\text{router}}} + \delta_{\text{max}}, \tag{9}$$

where  $\Theta_{\mathrm{router}}$  denotes the router throughput (bits/s). We compute the per-router traffic T(v) (bits) by accumulating all packets that traverse router v under the deterministic routes above, including transit traffic. The peak load is  $T_{\mathrm{max}} = \max_{v \in V} T(v)$ , which captures the worst-case congestion/hotspot for the given processing paradigm and topology. The maximum path propagation delay  $\delta_{\mathrm{max}}$  is defined as

$$\delta_{\text{max}} = L_{\text{n}} \cdot H_{\text{n}} + L_{\text{d}} \cdot H_{\text{d}} + L_{\text{w}} \cdot H_{\text{w}} + t_{r} \cdot (H_{\text{n}} + H_{\text{d}} + H_{\text{w}}), \quad (10)$$

where  $L_{\rm n}$ ,  $L_{\rm d}$ , and  $L_{\rm w}$  are the per-hop delays for intra-node, inter-die, and inter-wafer links, respectively, and  $H_{\rm n}$ ,  $H_{\rm d}$ , and  $H_{\rm w}$  are the corresponding hop counts along the longest communication path. The term  $t_r$  accounts for the router forwarding latency incurred at each hop. We calibrate  $L_{\rm n}$ ,  $L_{\rm d}$ , and  $L_{\rm w}$  using the prototype measurements in Table I, and take  $t_r$  and  $\Theta_{\rm router}$  from the modeled configuration in Table I. Overall, this model captures coarse traffic-induced serialization at the busiest router and topology-dependent propagation delay.

