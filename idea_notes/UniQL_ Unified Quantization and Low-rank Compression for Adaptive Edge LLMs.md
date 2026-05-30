## UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- baseline方法是什么？
  **Baseline 1（结构化剪枝）**：MoDeGPT（Lin et al., 2025）和SVD-LLM（Wang et al., 2025b）。
  - MoDeGPT：使用伪逆（Moore-Penrose inverse）在FP64精度下对MLP中间激活的通道相关性矩阵求解，排序权重通道以最小化剪枝误差。每个剪枝率需要重新计算伪逆（因为(W')^† ≠ (W^†)'），复杂度O(n³)，在Llama-3.1-8B的D_int=14336矩阵上伪逆耗时20.58分钟。对Qwen-2.5-7B（D_int=18944）因病态条件矩阵导致严重精度下降。
  - SVD-LLM：对权重矩阵做SVD分解后截断特征值，每次仅支持单一剪枝率，需为不同剪枝率独立运行。截断后需FT恢复精度。对于多个压缩率需多次独立运行（O(n)复杂度）。

  **Baseline 2（PTQ）**：TRT-AWQ（TensorRT-Model-Optimizer中的AWQ实现）和TAO-HQQ（TorchAO中的HQQ实现）。均为W4A16 PTQ框架，但仅支持固定4-bit量化，不支持结构化剪枝，且embedding/output层保持FP16（占用更大内存）。
  
  **全栈执行例子（Baseline MoDeGPT + AWQ在Llama-3.1-8B边缘部署）**：
  - 算法层：云上对MLP用伪逆排序（FP64, 20min/层），固定25%剪枝率 → 权重截断后精度下降（64.9% avg acc）→ GPTQ量化到W4A16。
  - 系统框架层：每个剪枝率需单独跑完整流程，生成多个模型副本存储（FP16 25%剪枝: 13.9GB per variant）。
  - 编译框架层：论文未明确说明。
  - Kernel调度层：使用标准INT4 GEMM kernel（Marlin），无融合RoPE优化。
  - 硬件架构层：部署到Nano 8G时仅使用固定尺寸的量化模型，无法根据设备当前内存使用动态调整。若设备负载高导致可用内存不足，需要换用更小模型或重新压缩。
  - 痛点总结：①伪逆计算慢且需FP64；②一个流程只产出一个压缩率；③无设备端自适应能力；④PTQ框架不支持SSM/Hybrid模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **UniQL方案**：用ridge leverage scores替代伪逆（复杂度降至O(n²)），用量化感知SVD融合Σ到U来降低量化误差，用masked LoRA一次训练多剪枝率，实现云侧一次压缩+设备侧自适应剪枝。

  **缺陷→方案映射**：
  1. 伪逆慢且不稳定 → 伪逆无关的ridge leverage score排序（Algorithm 1），在BF16下完成，Llama-3.1-8B MLP排序从7h03m降至19min（22×加速），且对Qwen-2.5-7B的大D_int/小D_h比不再失效。
  2. 每个压缩率需独立运行 → 一次masked LoRA微调（随机采样P_t），产出单个模型副本支持0%-35%所有剪枝率。压缩时间O(1) vs baseline O(n)。
  3. SVD分解在4-bit下引入量化误差 → 量化感知SVD（QSVD）：将W=UΣV分解后把Σ融合到U（W=(UΣ)V），使σ_i成为每列的量化scaling factor，避免长尾特征值被量化截断，4-bit 25%剪枝下精度提升7.5%。
  4. 不支持SSM/Hybrid模型 → 状态感知SSM权重排序：B-C排序考虑输入依赖的离散化广播外积，z-x-o排序从SSM状态H收集相关性。
  5. 设备端无自适应能力 → 部署INT4全量化模型（head-to-toe 4-bit，含embedding/output层），设备端在线解包→剪枝通道→重打包，支持按当前系统负载动态选择0-35%剪枝率。

  **全栈执行例子（UniQL端到端流程）**：
  - 算法层：校准集（Alpaca 128 samples, seq_len=2048）→ MLP ridge leverage scores排序 + MHSA量化感知SVD + Mamba状态感知排序 → 所有模块已排序 → Masked LoRA（r=8, Alpaca, 5 epochs, 每步随机采样P_t）→ GPTQ W4A16全局量化（含embedding/output层）→ 单个4-bit模型文件产出。
  - 系统框架层：单次云端压缩（A6000, 7h43m总耗时含FT+PTQ for Llama-3.1-8B），产出模型4.1GB → 推送到边缘设备。设备端无额外训练/压缩开销，仅做轻量级通道裁剪。
  - 编译框架层：论文未明确说明。
  - Kernel调度层：融合RoPE kernel（对称索引gather+slicing+旋转在单kernel完成，减少10%延迟）→ 设备端在线INT4解包→裁剪→重打包→送入Marlin 4-bit GEMM。Nano 8G上Qwen-2.5-7B的TPOT从TAO-HQQ的133.6ms降至77.2ms（1.7×）。
  - 硬件架构层：Jetson Orin Nano 8G统一内存。当OS报告高内存压力时，应用层触发p=25%或35%剪枝，模型从4.1GB降至3.2GB或2.8GB；当资源充足时p=0%，享受最高精度但最小延迟（2.7×-3.4× throughput vs FP16）。
  - 关键量化效果：Llama-3.1-8B在15%剪枝+4-bit下维持71.4% avg acc（仅比FP16全精度低2.6%），同时模型尺寸4.7×压缩（16GB→3.4GB），生成吞吐量3.4×提升。
