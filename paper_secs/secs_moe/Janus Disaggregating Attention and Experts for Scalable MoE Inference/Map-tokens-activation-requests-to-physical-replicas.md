# Map tokens' activation requests to physical replicas

12: **for** i = 1 to T **do** 

13: **for** j = 1 to k **do** 

14:  $O(i, j) \leftarrow \operatorname{actRep}[L(i, j)]$ 

experts to the least-loaded instances among those hosting their replicas (lines 2-11). This yields a near-balanced expert activation across instances while incurring only negligible computational overhead.

Synchronization-free scheduling. To avoid the overhead of global coordination, JANUS makes AEBS synchronizationfree across MoE instances via two mechanisms. First, JANUS implements AEBS as a GPU kernel to achieve microsecondlevel scheduling latency. This avoids CPU-GPU synchronization when accessing the per-token top-k routing results, and allows many tokens to be processed in parallel (i.e., steps 1 and 3 in Fig. 7). Second, JANUS trades a small amount of redundant computation to eliminate cross-instance synchronization. Instead of using a centralized global scheduler, each MoE instance independently runs the same AEBS kernel with identical input, including token activation patterns, replica layout, and instance metadata. Since AEBS is deterministic with respect to these inputs, all instances compute the same global assignment from logical experts to physical replicas. JANUS updates metadata such as replica layout only when the MoE sub-cluster is reconfigured, which occurs at a much coarser time scale (e.g., on the order of hours) than per-layer execution, making the propagation overhead negligible (§3.5). The redundant scheduling computation on each GPU is also small compared with the MoE forward computation. As a result, JANUS eliminates inter-GPU communication for activation scheduling while preserving correctness and imposing negligible overhead.

