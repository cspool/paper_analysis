# *A. Hardware Configuration*

We simulate an 8-GPU system interconnected via four NVSwitch units, replicating the topology of the NVIDIA DGX-H100 [39]. To enable accurate modeling, we extend Accel-Sim [23] with Hopper-specific architectural features and configure the GPU parameters based on the NVIDIA H100 specifications [36]. For multi-GPU communication, we integrate Accel-Sim with a customized BookSim2 [20], enabling

| Name        | Hidden<br>Size | FFN Hidden<br>Size | Attention<br>Heads | Sequence<br>Length | Batch<br>Size |
|-------------|----------------|--------------------|--------------------|--------------------|---------------|
| Mega-GPT-4B | 2048           | 8192               | 24                 | 1024               | 16            |
| Mega-GPT-8B | 3072           | 12288              | 32                 | 1024               | 12            |
| LLaMA-7B    | 4096           | 11264              | 32                 | 3072               | 3             |

TABLE I: LLM Settings Used in Evaluation.

concurrent execution across GPUs connected through a switchbased interconnect.

We further modify both Accel-Sim and BookSim2 with custom extensions to support the multimem instructions of NVLS. Specifically, following NVIDIA's NVLS design [24], we augment the "router" in BookSim2 to support in-switch multicast and reduction operations, and extend Accel-Sim to handle the translation from multimem addresses to virtual addresses at the Hub. The quantitative validation of our NVLS simulation is detailed in Section V-E. For fair comparison, we also augment T3 [43] with NVLS support by adopting the DMA-based NVLS design proposed by NVIDIA [24].

The NVLink and NVSwitch are modeled using real device parameters. NVLink is configured with a 16B flit size, a single-flit header, and bidirectional data transfer. NVSwitch employs round-robin arbitration with a 40 KB per-port Merge Table (320 entries) and supports routing to forward requests to their target GPUs. Each input port provides eight 256-depth virtual channels. We implement intra-SM request coalescing, aggregating multiple 32B sector requests into packets of up to 128B to emulate NVLink's burst transfer behavior. Link latency between GPUs and switches (from GPU to switch or from switch to GPU) is configured to 250 ns, resulting in a round-trip latency of approximately 1 µs.

