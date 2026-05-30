## Activation Offloading for MoE Model Compression（MoE模型压缩中的激活卸载）

术语是什么？
Activation Offloading for MoE Model Compression 是 QMoE 提出的一种 CPU-GPU 内存管理策略，用于在压缩万亿参数 MoE 模型时将 calibration 数据的中间激活存储在 CPU RAM 中，仅将当前需要处理的小块数据加载到 GPU。核心动机：data-dependent quantization（如 GPTQ）需要为模型中的每个 expert 收集足够的校准数据激活——对于有 2048 个 expert 的 c2048，需要 160K+ calibration samples，所有层的中间激活同时存储将需要数百 GB 显存，远超单 GPU 容量。Activation offloading 通过类似于推理时的 CPU offloading 反向操作（计算在 GPU、存储 bulk 在 CPU），使单卡 A6000 (48GB) 可压缩 3.2TB 模型。

从系统架构角度拆解术语：
**Dense Part 的执行流程**（每 Transformer block）：
```
// B: CPU RAM 中的大 buffer（list buffer 结构）
// GPU: 仅 hold 当前处理的小块数据 + 当前 expert weights

# Dense layers 处理
for each sample X (few hundred tokens) in B:
    1. Fetch X from CPU → GPU (PCIe)
    2. X_gpu → DenseLayers → Y_gpu
    3. Record expert assignments for each token in Y
    4. Send Y_gpu back to CPU, overwrite X in B (原地更新)
```

**Sparse Part 的执行流程**（每 Transformer block, per expert group）：
```
for each expert group E (|E|=16):
    1. Lazy fetch W_E from disk → GPU (如没在 GPU)
    2. Fetch X_E from CPU B → GPU（query by list buffer delimiter indices）
       // X_E = {所有分配给 E 中各 expert 的 token 的 hidden states}
    3. Batched GPTQ: compress all experts in E → E'_compressed
    4. Forward: Y_E' = E'_compressed @ X_E（使用压缩后权重）
    5. Send Y_E' to CPU, overwrite X_E in B
    6. Write compressed E' to disk, free GPU memory
```
关键约束：(1) 每个 token 每 Transformer block 仅读写 2 次（dense→Y, sparse→Y_E'），最小化 PCIe 传输；(2) 原地更新（overwrite）避免 CPU buffer 膨胀；(3) list buffer 数据结构的 delimiter-based 索引支持 O(1) per-sample access 和向量化 expert token 查询。

术语一般如何实现？如何使用？
- 实现：PyTorch CPU tensor buffer + CUDA stream 管理异步 CPU↔GPU 传输
- 内存需求：单 A6000 (48GB) GPU + few hundred GBs CPU RAM + 3.2TB disk（原始模型）
- Expert 分组大小 |E|=16 为 GPU memory-consumption vs utilization trade-off
- 通用性：可推广到任何需要处理 "模型太大、activation 太多、单卡放不下" 的 data-dependent 压缩/分析任务

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
