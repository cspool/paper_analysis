## Cross-Device Heterogeneous Pipeline for Speculative Decoding（投机解码跨设备异构流水线）

术语是什么？通过联网搜索让回答具体和精准。
Cross-Device Heterogeneous Pipeline是DFVG提出的FPGA-GPU异构speculative decoding流水线调度机制。核心思想是将speculative decoding的draft和verify两阶段分别部署在FPGA和GPU上并重叠执行：FPGA持续生成draft tokens（即使部分将被rejected），GPU从不idle（either verifying or forwarding），通过interrupt-driven coordination和PCIe异步通信实现tightly-coupled overlapped execution。该设计的理论基础是speculative decoding的非对称性：draft model轻量、latency-sensitive、bandwidth-intensive（适合FPGA低延迟streaming计算），verify model重量、compute-intensive（适合GPU高吞吐tensor operations）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DFVG的跨设备pipeline调度五阶段流程：
```
Phase 1: FPGA Draft
  FPGA draft model生成多分支candidate tokens (含动态长度)
  → 写入shared host buffer (BAR空间)
  → 更新status register
  → raise interrupt to CPU

Phase 2: GPU Verify (与Phase 1并行)
  CPU触发DMA → GPU从同一buffer读取candidates
  → TreeSort-Verify做block-parallel attention
  → acceptance decision via Eq.(3)
  → 返回accepted prefix

Phase 3: GPU Forward (若FPGA未完成)
  若FPGA仍在生成而GPU已完成verify:
  → GPU从accepted prefix继续forward生成新tokens
  → 直到FPGA请求新一轮validation

Phase 4: FPGA Rollback (若token rejected)
  FPGA对比GPU返回的accepted prefix length vs local sequence
  → 若rollback: reset KV cache到verified prefix
  → resume从verified prefix继续draft

Phase 5: Pipeline Continuation
  FPGA持续producing新drafts, GPU永不休眠
```
通信仅传输compact token IDs + status metadata（16 bytes/token：4B token ID + 4B confidence + 8B alignment），PCIe Gen4×16 (64GB/s)满足75%-85% acceptance rate下的data transfer需求。通信overhead占wall time仅1.08%-3.2%，不成为throughput bottleneck。V80 FPGA runtime功耗仅75W，使energy efficiency额外提升约1.7×（over GPU-only speculative decoding）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DFVG实现：CPU host controller (C++) + FPGA Verilog HDL + GPU CUDA 12.1。跨设备通信使用PCIe Gen4×16 (64GB/s) + shared host CPU memory ping-pong buffering。Xilinx XRT 2024.1管理FPGA-PCIe通信。流水线通过interrupt-driven coordination实现全异步操作。Ablation study中Pipe-Overlap贡献3.08×→3.26×加速。开源：https://github.com/ShaoqiangLu/DFVG。该架构与DuoDec (CPU+GPU) 和Dovetail (GPU+CPU) 形成对比：DFVG首次引入FPGA用于draft stage并实现overlapped execution，相比DuoDec的1.67×和Dovetail的1.43×，达到3.26× speedup。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU
