# B Extending to the CUDA Ecosystem

Although ElasticMoE is implemented on Ascend NPUs, the framework can be readily extended to the CUDA ecosystem. We have developed a barebones proof-of-concept implementation on NVIDIA GPUs that confirms the feasibility of this port.

<span id="page-16-2"></span>![](_page_16_Figure_0.jpeg)

Figure 12. Scale-down latency comparison across baseline methods for three MoE models. The x-axis indicates scaling configurations, represented as source  $\rightarrow$  destination NPU transitions, corresponding to fixed step size for (a) and (b), and progressively larger steps for (c). Similar to the scale-up scenario, in all cases, ElasticMoE consistently achieves substantially lower latency than competing baselines. In our approach, the shaded purple region denotes the warm-up time.

On the *HMM side*, the control-plane logic remains unchanged, since resource tracking, scaling orchestration, and zero-copy coordination are device-agnostic. The data plane, however, must replace CANN-specific primitives with CUDA equivalents. For example, CUDA provides cudaIpcGetMem Handle, cudaIpcOpenMemHandle, and cudaMalloc for interprocess memory sharing, which substitute Ascend's IPC APIs. Similarly, virtual expert management can be supported through CUDA's virtual memory primitives [16], such as cuMemAddressReserve and cuMemMap, which enable pagebased allocation and remapping.

On the *IMM side*, the design also remains largely unchanged. The same zero-copy loader and instance manager can be reused, with the only difference being the backend: instead of ascend-vLLM, the implementation would rely on the standard vLLM for CUDA-enabled GPUs.

Overall, extending ElasticMoE to CUDA primarily involves swapping low-level device APIs, while the higher-level control, coordination, and inference logic remains intact. This demonstrates that ElasticMoE's design is portable across accelerator ecosystems with minimal changes.

#### <span id="page-16-1"></span>C Limitations and Future Work

While ElasticMoE demonstrates the feasibility and benefits of fine-grained, zero-downtime vertical scaling, several limitations remain.

First, although the system supports fine-grained scaling through adjustments in DP and EP degrees, the TP degree needs to be held fixed. This restriction simplifies migration by keeping shared model weights and KV cache layouts unchanged, but it also constrains the granularity of scaling. For some configurations high TP (although rare), the minimum scaling unit remains tied to TP size, limiting elasticity and granularity.

Second, ElasticMoE eliminates downtime by keeping the active instance serving requests while the new configuration is prepared in parallel. However, during the transition, the

active instance pauses intake of new requests, which reduces effective batch size and temporarily lowers throughput (Table 2). While preferable to downtime, this reduced capacity highlights a trade-off between availability and performance during scaling.

**Future work.** Building on the above-mentioned limitations, one promising direction is to relax the fixed-TP constraint. Supporting flexible TP degrees would allow even finer scaling but introduces new challenges, such as complex sharding, migrating weights and reshaping KV caches without incurring high latency or service interruption. Developing techniques to manage this additional complexity while keeping scale-up latency small remains an open systems problem.

Another avenue is to improve transition capacity. Currently, the active instance operates at reduced throughput during scale-up. A more ambitious design would keep the old instance at full capacity until the new one is ready, or even allow both instances to serve requests concurrently sharing compute resources. Realizing this vision would require mechanisms to manage two independent activation spaces, coordinate KV cache block allocation across instances, and safely migrate requests in-flight—posing interesting challenges for distributed memory management and scheduling.

