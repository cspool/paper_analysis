# VI. DISCUSSION

Storage Overhead of SR-Based Expert Compression. The additional storage overhead introduced by our proposed

expert compression algorithm can be handled. Specifically, it consists of expert residual and shared expert. ① The expert residual consumes little GPU memory due to its high compressibility. ② The shared experts compete with local experts for GPU memory, which can be solved by offloading local experts to CPU memory while keeping shared experts in GPU memory. Offloading local experts to CPU memory is an effective strategy, which has been well studied (e.g., Zero-Offload [44]) and can be directly integrated into HybridEP.

Backward Propagation in MoE Training. The backward phase has the unique All-Reduce communication to synchronize model parameters, which competes with other types of communications and thus affects our modeling. Nevertheless, our modeling is still effective for backward propagation because the All-Reduce communication traffic is relative to the model size, and its latency can be regarded as a constant when model configurations are determined. Therefore, our modeling can handle backward propagation by simply adding a constant.

#### VII. RELATED WORKS

We introduce works that are orthogonal (or related) to our study, which mainly focus within the high performance cluster.

**Optimizations on the Gate Network.** Our modeling assume that the gate network activates experts evenly, and many works focused on how to achieve this. For example, Lewis et al. [29] proposed the BASE layer with token-to-expert allocation schema. Zhou et al. [60] proposed to allow experts to choose tokens. HybridEP can integrate them.

**Optimizations on A2A Communication.** To reduce A2A time, existing works focus on improving bandwidth utilization and reducing communication volume. For example, Hetu-MoE [38] proposed a hierarchical A2A algorithm to reduce communication rounds of inter-node communications; [22], [42] proposed the 2D-hierarchical A2A algorithms to better utilize high-speed intra-node links; Zhou et al. [59] used ZFP compression to reduce the A2A traffic.

**Optimizations on Comp. & Comm. Overlap.** HybridEP combines prefetch and pipeline to fully overlap computation and communication, while existing studies try to optimize one of them as much as possible. For example, [22], [30], [46] try to find the optimal pipeline degree to fully overlap expert computation and A2A communication, while Janus [35] tries to increase overlap time by pre-fetching experts.

