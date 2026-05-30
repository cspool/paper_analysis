# X. RELATED WORK

*DRAM-Centric General-Purpose Accelerators:* Systems such as NVIDIA H100 [14], AMD MI300x [61], SambaNova [52], and TPU [26] use high-capacity HBM, large shared caches, and dense compute to support both training and inference. These architectures typically feature a single NUMA domain with distributed controllers and centralized caches. While this enables flexible data access, it creates long memory paths and high energy per access, which is especially harmful for memory-bound decode. In contrast, the RPU uses a fine-grained NUMA design where each core has its own HBM-CO DRAM channel and local SRAM buffer, eliminating shared caches and reducing on-chip data movement. This decouples compute, memory, and network pipelines, sustaining high bandwidth utilization.

*SRAM-Centric Custom Accelerators:* Custom accelerators such as Groq [2], Cerebras WSE-3 [64], and Graphcore IPU [32] rely on SRAM as main memory. However, the limited density of SRAM makes it impractical to store large models efficiently. For example, a 70B parameter model deployed on Groq requires hundreds of accelerator cards, while Cerebras spans four wafer-scale chips.

To utilize their full SRAM bandwidth, these systems shard each matrix across a large compute fabric. For example, Cerebras may distribute a single VMM across 900,000 cores, requiring vector broadcasts to traverse up to 1,000 core-to-core hops and reductions to span the entire wafer. With the model globally distributed, network communication, not compute or memory access, becomes the primary performance bottleneck due to SRAM's low density. In contrast, each RPU reasoning core is significantly more capable than the ultra-lightweight cores used in Groq or Cerebras, with higher FLOP throughput and wider data buses. As a result, more of the workload is processed locally per core, reducing reliance on multihop communication. Additionally, the RPU's hierarchical ring network has a much smaller diameter than mesh or waferscale fabrics, further minimizing the number of hops required for vector broadcasts and reductions.

*Processing In Memory (PIM) Accelerators:* PIM architectures aim to reduce data movement by embedding compute capabilities within or near memory [13], [24], [36], [46], [70]. Many PIM designs leverage DRAM or emerging memory technologies to perform simple operations, typically integer or bitwise logic, in situ. While effective for low-intensity workloads, PIM designs struggle when arithmetic intensity

|                                 | System Metrics |                 |                       | Speculative Decoding  |            |                   |                    |
|---------------------------------|----------------|-----------------|-----------------------|-----------------------|------------|-------------------|--------------------|
| System                          | Main<br>Memory | BW/Cap<br>(1/s) | Comp/BW<br>(Ops/Byte) | Systems<br>(Spec-70B) | TDP<br>(W) | Shoreline<br>(mm) | Perf<br>(Tokens/s) |
| NVIDIA H200                     | HBM3e          | 34 ↓            | 206 ↑                 | 1 -                   | 700 ↓      | 66 ↓              | 134 ↓              |
| SambaNova                       | HBM3           | 25 ↓            | 399 ↑                 | 16 -                  | 10k ↑      | 704 ↓             | 457 ↓              |
| Groq LPU                        | SRAM           | 355k ↑          | 2.4 ↓                 | ‡<br>500 -            | 100k ↑     | NA .              | 1678 ↓             |
| Cerebras WSE-3                  | SRAM           | 477k ↑          | 6.0 ↓                 | 4 -                   | 136k ↑     | NA .              | 2148 ↓             |
| RPU                             | HBM-CO         | 500 .           | 32 .                  | 200CU -               | 1.8k .     | 1500 .            | 4423 .             |
| ‡ Groq est. 400-600 Processors. |                |                 |                       |                       |            |                   |                    |

Fig. 14. A comparison of leading hardware platforms. Speculative decoding throughput for Llama3-70B based on published data [2], [52], [57], [64].

exceeds 1 Op/Byte, which is common during LLM inference. PIM architectures are also poorly suited for floating-point operations or fine-grained programmability to support rotary embeddings, softmax, and normalization functions.

Furthermore, the rise of block-quantized formats (e.g., BFP, MXFP) poses a major challenge for PIM. These formats require dynamic exponent broadcasting, alignment, and decoding before arithmetic. These steps involve conditional logic and variable indexing, which are difficult to implement in DRAM-compatible circuitry.

*Comparison Under Speculative Decoding:* Speculative decoding is an increasingly common technique used in LLM inference to reduce token generation latency by leveraging a lightweight "draft" model to predict multiple tokens ahead. These predicted tokens are then validated by a larger "target" model; if the predictions are correct, several tokens can be committed in parallel. This approach may be challenging because it increases the arithmetic intensity of each query.

Industry accelerators often report performance under speculative decoding. We evaluate the RPU using a comparable speculative decode setup. In our evaluation, we adopt an 8 token lookahead configuration in which a Llama3-8B draft model proposes tokens for a Llama3-70B target model. On average, 4.6 tokens are accepted per speculative window [41], accelerating end-to-end inference by 1.8×. Figure 14 compares our speculative performance to publicly reported numbers from NVIDIA H200 [57], SambaNova SN40L [52], Groq [2], and Cerebras WSE-3 [64]. The RPU-200U configuration is lower latency than all evaluated systems.

