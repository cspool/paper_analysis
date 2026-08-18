## LLMCompass（LLM 推理 cycle-accurate 模拟器，ScaleSim 的 LLM 扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LLMCompass 是 Princeton 在 ISCA 2024 提出的 LLM 推理硬件评估框架（https://github.com/PrincetonUniversity/LLMCompass，Python、基于 ScaleSim 的 cycle-accurate systolic array 模拟器）：利用 LLM 计算图中 dense operator（GEMM/GEMV、softmax、LayerNorm）结构化和可预测的 compute/memory access pattern，在 block/tile 粒度快速评估硬件设计，含自动 mapper 生成性能最优的 mapping/schedule 与面积成本模型，对算子平均延迟误差约 10.9%、完整 LLM 推理约 4.1%。SMOOTH（ISCA'26）把它作为评估平台：论文称"LLMCompass, an LLM-optimized extension of ScaleSim"，在其上集成 end-to-end SRAM 管理器模拟 block 分配、early reclamation 与预取。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子（SMOOTH 配置）：输入 transformer 模型的结构元数据（TinyLLaMA/GPT-Neo/Gemma-2/LLaMA2/Bloom/GPT-3 的层数、d_model、head、w4a8/int8 量化位宽）+ 移动 NPU 硬件配置（Table III：940MHz、1 core、32×32 Matrix Engine、32-lane Vector Engine、2/8/32MB SRAM、16/32/64/128GB/s DRAM、batch=1）→ LLMCompass 调用 ScaleSim 对每算子做 systolic array cycle 级时序仿真 → SMOOTH 的 SRAM manager 在其中建模 block table/bitmap 分配、end_cmd 回收、N_preload 预取 → 输出 TTFT/TTLT/ITL、SRAM occupancy、hit tile 比例、能量 → 归一化绘图（fig14/16/20）。因用结构元数据而非真实权重，无需大权重数据集，AE 约 20 小时/48 核、磁盘约 10GB。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用方式（LLMCompass 官方 + SMOOTH AE）：`git clone https://github.com/PrincetonUniversity/LLMCompass`（ISCA_AE 分支），依赖 Python 3.9、PyTorch 2.0.0、scalesim==2.0.2、matplotlib/seaborn，可用 Docker；SMOOTH 复现流程 `cd src/policies && bash run_all_policies.sh` 跑各策略仿真、`src/ae/figure14|16|20` 的 plot 脚本出图、`src/verilog/run_all.sh` 用 Yosys 综合 RTL。知识库已有独立 SCALE-Sim 条目（知识库_硬件架构.md），LLMCompass 是其在 LLM 场景的框架化扩展。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference
