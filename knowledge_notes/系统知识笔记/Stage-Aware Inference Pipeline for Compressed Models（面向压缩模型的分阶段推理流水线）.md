## Stage-Aware Inference Pipeline for Compressed Models（面向压缩模型的分阶段推理流水线）

术语是什么？通过联网搜索让回答具体和精准。
Stage-aware inference pipeline是ZipServ提出的一种根据推理阶段（prefill vs decode）切换执行策略的系统设计。其核心观察是：prefill阶段是compute-bound（高算术强度，大batch矩阵乘法），decode阶段是memory-bound（低算术强度，每token产生少量FLOPs但需加载全部权重）。因此ZipServ在decode阶段使用fused ZipGEMM kernel（load-compressed, compute-decompressed），消除中间buffer，最大化compute intensity；在prefill阶段使用decoupled pipeline（先decompression kernel解压到global memory，再cuBLAS_TC GEMM），利用prefill的高算术强度摊销解压overhead（<4% total time overhead）。这种动态策略选择避免了"一刀切"导致的某一阶段性能退化。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（ZipServ+vLLM）：
1. 请求到达vLLM scheduler → 进入prefill phase
2. Prefill阶段：scheduler检测到large N（N=BS×SeqLen，如8192 token）→ 选择decoupled pipeline → 发射Decompression kernel解压TCA-TBE权重到global memory → 发射cuBLAS_TC GEMM（高吞吐）→ attention计算 → KV cache写入
3. 进入decode phase（N=BS×1，如batch 32则N=32）→ scheduler切换到fused ZipGEMM → 每个decode step直接发射ZipGEMM kernel，从DRAM读压缩权重→register内解压→Tensor Core mma计算→输出token
4. 压缩权重节省的GPU memory（LLaMA3.1-8B: 14.96→10.83GB, 节省3.78GB）自动分配给KV cache（5.07→8.60GB），通过vLLM PagedAttention管理
5. 系统整体效果：平均1.22× end-to-end加速over vLLM，memory saving 25-29%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为vLLM的PyBind11 extension（约1.0K行Python glue code）。关键实现点：
1. Model loader修改：识别TCA-TBE压缩格式，加载压缩权重buffer
2. Linear execution module修改：根据当前推理阶段（prefill vs decode）选择kernel path
3. 切换逻辑：在decode iteration中始终用ZipGEMM；prefill由batch size和seq_len计算N，若N>阈值（如128）则走decoupled path
该设计理念可推广至任何有"压缩-计算"tradeoff的场景（如KV cache compression、activation compression），根据workload特性动态选择最优执行策略。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression
