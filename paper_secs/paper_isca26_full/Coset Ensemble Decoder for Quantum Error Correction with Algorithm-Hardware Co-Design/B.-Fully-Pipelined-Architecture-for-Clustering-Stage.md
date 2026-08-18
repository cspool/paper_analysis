# *B. Fully-Pipelined Architecture for Clustering Stage*

These stalls primarily stem from RAW (Read-After-Write) hazards during sequential growth and memory bank conflicts under concurrent updates, motivating our specialized 7-stage pipeline in Fig. [5](#page-6-0) with targeted mechanisms to maximize hardware utilization.

The 7-stage pipeline processes one Vertex ID (VID) per cycle to achieve high-throughput decoding. To handle concurrent merge operations, we introduce a hierarchical ID mapping that interposes a Root-ID (RID) between VIDs and Cluster-IDs (CIDs). This indirection layer decouples physical vertex storage from logical cluster states. The dataflow operates as follows: (S1–S3) The pipeline dequeues a VID and concurrently fetches its associated RIDs and edge weights to resolve CIDs. (S4) Grow/merge logic is evaluated based on the retrieved metadata. (S5–S7) The pipeline manages active CIDs via a priority-based FIFO and updates boundary-vertex states. As detailed in Sec. [IV-D,](#page-6-1) this RID-based hierarchy is key to collapsing write fan-out and mitigating peak memory bandwidth pressure.

To address the identified bottlenecks, we implement two core optimizations: (1) a forwarding/bypass network that feeds S4's growth decisions back to earlier stages to resolve data dependencies, and (2) a hash-based memory layout that minimizes contention during concurrent metadata lookups.

# *B. Fully-Pipelined Architecture for Clustering Stage*

These stalls primarily stem from RAW (Read-After-Write) hazards during sequential growth and memory bank conflicts under concurrent updates, motivating our specialized 7-stage pipeline in Fig. [5](#page-6-0) with targeted mechanisms to maximize hardware utilization.

The 7-stage pipeline processes one Vertex ID (VID) per cycle to achieve high-throughput decoding. To handle concurrent merge operations, we introduce a hierarchical ID mapping that interposes a Root-ID (RID) between VIDs and Cluster-IDs (CIDs). This indirection layer decouples physical vertex storage from logical cluster states. The dataflow operates as follows: (S1–S3) The pipeline dequeues a VID and concurrently fetches its associated RIDs and edge weights to resolve CIDs. (S4) Grow/merge logic is evaluated based on the retrieved metadata. (S5–S7) The pipeline manages active CIDs via a priority-based FIFO and updates boundary-vertex states. As detailed in Sec. [IV-D,](#page-6-1) this RID-based hierarchy is key to collapsing write fan-out and mitigating peak memory bandwidth pressure.

To address the identified bottlenecks, we implement two core optimizations: (1) a forwarding/bypass network that feeds S4's growth decisions back to earlier stages to resolve data dependencies, and (2) a hash-based memory layout that minimizes contention during concurrent metadata lookups.

