# *C. Communication in LLM Inference*

Collective communication is widely used in model training and inference for data synchronization across parallel patterns. Taking TP as an example, the activation synchronizations at the end of each MLP and attention layer introduce All-Reduce to obtain the fully reduced output. Recent efforts are devoted to optimizing collective communication on mesh topologies through efficient algorithms and scheduling. MultiTree proposes a topology-aware link-scheduling approach that maps a tree algorithm at every node, achieving strong performance [26]. TTO improves link bandwidth utilization through chunk overlapping [37]. TidalMesh further pushes the performance boundary of All-Reduce on 2D meshes via the overlapping of Reduce-Scatter and All-Gather [42].

However, TTO removes nodes from the system to enforce a tree topology, while TidalMesh is specifically designed for a 2D mesh; both can fail to maintain high performance under certain fault conditions. MultiTree achieves stable performance across arbitrary topologies, including those with faults, but lacks global awareness of link contention, which can lead to performance degradation.

