## AiMX PIM Architecture

术语是什么？通过联网搜索让回答具体和精准。

AiMX是SK hynix基于GDDR6-AiM (Accelerator-in-Memory) 芯片构建的AI加速卡产品线，属于Processing-in-Memory（PIM）的商业化实现。GDDR6-AiM芯片在标准GDDR6 DRAM的每个bank内嵌入MAC（Multiply-Accumulate）和AF（Activation Function）计算单元，使内存芯片本身能够执行计算。AiMX卡将多个GDDR6-AiM芯片组合到PCIe加速卡上——2024年升级版为32GB容量，支持Llama 3 70B等大型模型的推理。2025年AI Infra Summit展示了AiMX与NVIDIA H100 GPU的disaggregated inference系统（2×H100 + 4×AiMX），通过vLLM框架支持长token生成和推理模型。核心规格：1ynm工艺，8Gb per chip density，16Gb/s/pin GDDR6接口，1.25V电压（低于标准1.35V），单片1 TFLOPS MAC peak compute，每module 16 banks配备并行MAC单元。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。

AiMX作为PIM加速器，在disaggregated LLM inference系统中承担memory-bound computation（主要是attention中的GEMV操作），GPU/NPU承担compute-bound computation（如GEMM/FC layers）。典型执行流程：(1) host CPU/GPU通过GDDR6 memory interface发送PIM ISR（Instruction Set Register）指令到AiM DRAM controller；(2) controller解析指令——MAC_ABK对所有16 bank同时执行256-bit MAC，MAC_SBK仅指定bank执行，WR_ABK/SBK写数据到bank，RD_ABK/SBK从bank读结果；(3) controller内的Global Buffer (GB)暂存中间结果，通过instruction-level channel_mask支持multi-channel操作；(4) 指令间通过SYNC barrier协调。在PIMphony论文中，AiMX被用作校准和验证baseline的PIM architecture reference——Ramulator-based simulator的DRAM command timing和resource contention按AiMX specification校准。论文提出的硬件修改（OBuf扩展、DCS controller logic、on-module dispatcher）均基于AiMX架构参数评估area/power overhead（OBuf 0.47% of MAC unit area/bank，DCS +0.5% area/+1.3% power on PIM HUB control blocks，dispatcher 4% area）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

商用AiMX通过PCIe卡形式部署，软件栈支持vLLM框架。学术研究使用Ramulator 2.0-based simulator（如github.com/arkhadem/aim_simulator）建模AiM架构，该simulator支持可配置的bank数、channel数、MAC单元数、GB大小和DRAM时序参数。PIMphony通过MLIR compiler生成PIM instruction sequences，经IREE runtime HAL对接commercial PIM SDK，在simulator上评估multi-node PIM部署。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

