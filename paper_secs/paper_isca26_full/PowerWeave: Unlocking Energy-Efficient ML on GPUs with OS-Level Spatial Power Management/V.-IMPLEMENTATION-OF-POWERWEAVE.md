# V. IMPLEMENTATION OF POWERWEAVE

By extending the LithOS GPU operating system [12], we implement a real-world prototype of PowerWeave's Interposer in ∼5500 lines of Rust and its Governor in ∼250 lines of Python. In this section, we describe the key implementation decisions behind PowerWeave.

Interposer. PowerWeave transparently intercepts all CUDA driver API calls on kernel launch paths, requiring no modifications to the application, serving framework, or GPU driver. On a kernel launch, the interposer records the kernel's function handle, grid and block dimensions, shared-memory size, and the CUDA stream. These fields together form the kernel's identity key used by the profiler. Kernel completion times are obtained by injecting CUDA event pairs around each launch and querying their elapsed time asynchronously, avoiding any blocking on the critical path. The profiler, predictor, and DVFS controller all execute within the interposer's address space on a dedicated background thread. Frequency changes are issued through the NVIDIA Management Library (NVML), which exposes per-GPU clock-setting interfaces.

Governor. The governor is implemented as a minimal library that can be imported into existing serving frameworks. Once imported, it monitors per-domain request rates and tail latencies against the configured SLO targets. On each control tick it estimates the acceptable performance degradation each domain can tolerate to preserve the SLO and communicates this slack to the interposer. Obtaining userspace-level metrics from frameworks such as vLLM or SGLang requires only a few lines of code. The governor's design is intentionally flexible: SLO targets, monitoring window duration, and the tail-latency percentile used for decisions are all configurable per domain, enabling the same core mechanism to support custom latency-driven, per-tenant, and throughput-balancing policies without code changes.

Compatibility with native GPU Sharing Mechanisms. PowerWeave's spatial partitioning model is compatible with existing GPU sharing mechanisms. NVIDIA MIG enforces hardware-level spatial partitions, and the governor can operate within MIG instances without modification. For MPS deployments, PowerWeave performs TPC assignments as in LithOS [12], which is based on MPS. Alternatively, spatial isolation within MPS can be achieved through NVIDIA Green Contexts [42]. AMD GPUs expose analogous mechanisms:

![](_page_7_Figure_0.jpeg)

Fig. 7: Additional Components for Per-SM DVFS granularity.

SPX/DPX/CPX modes on MI300X and MI355X [1] provide MIG-equivalent isolation across XCDs, and CU masking via ROCm [2] enables MPS-style fine-grained assignment of streams to Compute Units. Similarly, PowerWeave can adopt either path without changes to its core design.

