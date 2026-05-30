## Wafer-Scale Engine (WSE / 晶圆级引擎)

术语是什么？
Wafer-Scale Engine (WSE) 是 Cerebras Systems 推出的晶圆级 AI 加速器芯片。与传统的 multi-die GPU 不同，WSE 在整片 300mm 硅晶圆上集成计算核心，消除了芯片间通信（off-chip）的带宽瓶颈。WSE-2（第二代）使用 TSMC 7nm 工艺，在 46,225 mm² 面积上集成 2.6 万亿晶体管、850,000 个 AI 核心、40 GB on-chip SRAM 和 20 PB/s 内存带宽。系统形态为 CS-2（单 WSE-2 芯片的整机系统）。

从芯片设计角度拆解术语：
WSE-2 的芯片设计核心特征：
1. **单晶圆集成**：84 个 die 通过专有互联技术拼接在单晶圆上，构成统一的 2D mesh 网络。每个 PE 通过 32-bit 双向端口连接 4 个邻居（N/S/E/W），单跳延迟 1 cycle。
2. **分布式内存模型**：无共享内存、无硬件 cache coherence。每个 PE 有 48 KB scratchpad SRAM，所有数据传输由软件显式管理。这消除了 GPU 上常见的 cache miss 和 coherence 开销。
3. **数据流架构**：weight streaming 模式将参数存储与计算解耦——权重从片外 stream 到片内，激活在片上流转。支持任意规模模型的线性扩展数据并行训练。
4. **稀疏度收割**：数据流架构天然跳过零值计算，对稀疏 tensor 有效。
5. **与 GPU 的物理对比**：WSE-2（46,225 mm², 850K cores, 40 GB SRAM, 20 PB/s BW）vs NVIDIA H100（814 mm², 14,592 FP32 + 456 Tensor Core, 0.05 GB on-chip, 0.002 PB/s BW）。WSE-2 面积约 57×，核心数 56×，片上内存 800×，带宽 10,000×。

BTA 论文的关键洞察：WSE-2 的巨量片上带宽和近计算 SRAM 改变了 MoE 训练的瓶颈——GPU 上的通信瓶颈在 WSE-2 上得到缓解，但 attention 的激活内存（KV/softmax 中间结果）限制了 batch size 的上限，形成"attention 内存 vs expert 计算密度"的新瓶颈。

术语一般如何实现？如何使用？
Cerebras CS-2 系统通过 Cerebras Software Platform（包括 CGC 编译器、PyTorch 集成）提供编程接口。PyTorch 模型可直接在 CS-2 上训练/推理。weight streaming 机制将模型权重存储在外部 MemoryX 系统中，按需 stream 到晶圆上的 PE。BTA 论文中使用 Qwen3-30B-A3B 在 CS-2 上训练，通过配置 `ws_opt_enable_bta: true` 启用 BTA。WSE-2 的代表性性能：在 Llama 4 推理上达到 2,522 tokens/s（约 H100 的 2.5×）。

涉及论文标题：
- Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors
