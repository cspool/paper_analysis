## AttAcc 模拟器（GPU-PIM 异构 batched Transformer 推理模拟器）

术语解释
ASPLOS'24（SNU scale-snu 组）的 GPU + HBM-PIM 异构推理模拟器，CHIME 用它做 GPU 侧 roofline 模拟与 HBM-PIM 基线系统评估。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AttAcc 是面向 batched Transformer 生成式推理（TbGM）的 GPU-PIM 异构系统模拟器：顶层处理器模拟用 Python，DRAM 侧基于 ramulator2 类 C++ 实现；PIM 建模为 HBM3-PIM——bank 级 GEMV 单元 + buffer 级 softmax 单元，提供 PIM 命令（如 PIM_MAC_AB 全 bank GEMV、PIM_MAC_SB 同 bank GEMV、PIM_MAC_PB 逐 bank GEMV），配套自定义 frontend/mapper/controller 与 PIM 请求调度器；GPU 侧以 roofline 分析模拟 FC 计算。论文报告相对同容量传统系统最高 2.81× 性能与 2.67× 能效。开源：https://github.com/scale-snu/attacc_simulator。其 AFD 基线为"GPU 保留所有 HBM-PIM 存储 KV cache"，HBM-PIM 容量不足是该系统吞吐受限的根源（CHIME 的 DRM 分析对象之一）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME 的用法：把 AttAcc 作为 GPU 侧的 roofline 模拟器（FC 吞吐按 batch 变化建模），并把 GPU+HBM-PIM（640GB+260.8TB/s GPU 侧）与 HBM-PIM-EXT（320GB+130.4TB/s，同成本预算）作为 5 个基线中的 2 个，统一假设 attention 期间加速器带宽满利用；CHIME-PIM 侧则不用 AttAcc，而是自建 CHIME-PIM-sim（修改版 DRAMSim3）。模拟流程：请求 trace → 调度器组批 → GPU 侧 roofline 给 FC 延迟、PIM 侧逐 cycle 推演 → 合成端到端吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：roofline 近似 + cycle 级 PIM 时序组合，避免了全细节 GPU 仿真；开源仓库含 README 构建与运行说明。使用方式：评估 HBM-PIM AFD 系统（NeuPIMs/AttAcc 线）的吞吐/能效；其容量假设（HBM 全部给 KV）是保守上界，实际部署模型权重还要占用部分 HBM（CHIME 据此修正为 ~310GB）。注意 AttAcc 不模拟 DIMM-PIM（CHIME 需另建模拟器）。DCC（ISCA'26）进一步修改并扩展该模拟器：在 AttAcc 后端之外加入 Samsung HBM-PIM 后端（每两 bank 16-way FP16 FPU + 两个 16×256-bit GRF 的 DRAM 计算命令建模），平台配置 A100 + 5 个 HBM3 设备（5.2Gbps/pin、333MHz、16 pCH×64 bank；tCK=0.79、tRCD=19、tRP=19、tCL=19、tCCD=4、BL=2），数据搬移用 Ramulator 2.0 的 LD/ST 命令仿真，DCC 生成的 tiling draft 指令 trace 注入做 trace-driven 时序仿真。评测 GEMV/RED/ATTN（输入 128）与 GPT3-13B/LLaMA2-33B/MT-NLG-310B：DCC 较 AttAcc 默认实现平均提速 1.26×/1.48×/1.23×（GEMV/RED/ATTN），AttAcc_Full+DCC 对 GPU 端到端平均 4.52×（峰值 7.71×）。修改版模拟器随 DCC 仓库发布：github.com/SPIN-Research-Group/DCC（MIT，Zenodo 10.5281/zenodo.19442321）。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
