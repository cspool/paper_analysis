## DRAM-Based Processing-in-Memory (PIM)

术语是什么？通过联网搜索让回答具体和精准。

DRAM-Based Processing-in-Memory（PIM）是一种将计算逻辑集成到DRAM内存芯片内部的架构技术，通过在DRAM bank附近或内部放置计算单元（如MAC单元、ALU），利用DRAM内部极高的带宽（数百GB/s至TB/s级别）直接在内存中执行计算，避免数据在内存和外部处理器之间的昂贵搬运。与传统的冯·诺依曼架构（计算与存储分离）不同，PIM将计算带到数据所在位置，特别适合memory-bound的workload（如LLM decoding中的GEMV操作，compute intensity极低）。商业产品包括SK hynix的GDDR6-AiM/AiMX（1ynm工艺，8Gb density，16Gb/s/pin GDDR6接口，1 TFLOPS MAC/chip）和Samsung的HBM-PIM/LPDDR5X-PIM。研究原型包括UPMEM-PIM（通用可编程PIM，基于DDR4 DRAM芯片内嵌in-order DPU核心）和基于DRAM技术扩展的各种学术PIM架构。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。

DRAM-based PIM的芯片设计核心是在标准DRAM芯片的bank区域内或bank I/O路径上插入计算逻辑。以SK hynix GDDR6-AiM（ISSCC 2022）为例：芯片包含16个DRAM bank，每个bank配备专用MAC（Multiply-Accumulate）和AF（Activation Function）单元。PIM操作通过ISR（Instruction Set Register）指令从外部host controller发送到AiM DRAM controller，controller解析指令并生成DRAM命令序列。MAC_ABK（All-Bank MAC）：16个bank同时执行256-bit MAC操作——每个bank从自身存储阵列读取operand，送入bank-local MAC单元执行乘加，结果写回bank或Global Buffer（GB）。MAC_SBK（Single-Bank MAC）：仅指定bank执行MAC。这种bank-level parallelism设计的关键约束：(1) all-bank操作功耗峰值高（16 bank同时active）可能超过power budget；(2) DRAM bank原本设计为分时激活（一次只active少数bank），同时active所有bank违反标准DRAM时序假设。因此PIM芯片设计需要在标准DRAM物理约束（timing、power delivery、thermal）和计算并行度之间权衡。在PIMphony论文中，PIM module配置为16-channel × 16-bank，internal bandwidth达16-32TB/s（远超GPU HBM的~2TB/s），每个channel内的PIM controller/HUB管理命令调度和address translation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

商用PIM芯片（如SK hynix GDDR6-AiM）作为标准GDDR6内存的pin-compatible替代品，可集成到标准内存总线。AiMX卡将多个GDDR6-AiM芯片组合为PCIe加速卡（2024版32GB），通过vLLM等框架支持LLM推理中的attention offloading。学术研究（如PIMphony、CENT、NeuPIMs）通常使用Ramulator-based cycle-accurate simulator模拟PIM执行，结合AiMX specification校准DRAM command timing。开源AiM simulator（github.com/arkhadem/aim_simulator）基于Ramulator 2.0提供可配置的PIM建模。PIM编程模型通常为host-controlled：host发出PIM指令（如WR-INP写输入、MAC执行乘加、RD-OUT读输出），PIM controller在module内解码执行。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
- Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
