## RoMe: Row Granularity Access Memory System for Large Language Models

- 属于硬件架构的实现是什么？实验比较什么？
  提出RoMe硬件架构，将HBM4的MC-DRAM接口从32B column-level RD/WR改为4KB row-level RD_row/WR_row，包含三大组件：(1) Virtual Bank (VBA)：从MC可见接口中移除bank group和pseudo channel。一个VBA由两个不同bank group的bank组成，以time-multiplexed方式交错访问（tRRDS−tCCDS intentional delay插入后），两个pseudo channel并发工作，使单个VBA即可提供最大带宽。有效row size从1KB提升到4KB，banks/channel从128降到32。(2) Command Generator：放置在HBM logic die中，接收RD_row/WR_row后静态展开为固定DRAM命令序列（ACT→连续RD/WR→PRE），按tCCDS间隔对两个bank做精确交错。command generator不根据bank state动态发命令，而是按预定时序静态发出。(3) RoMe MC：只发出RD_row、WR_row、REF三种命令；bank states缩减为Idle/Writing/Reading/Refreshing四个；timing parameters从15个缩到10个；bank FSM数量从每PC所有bank缩到5个；request queue达到峰值吞吐仅需2个entries（HBM4需≥45）；调度简化为跨VBA交错+oldest-first公平性，无需page policy。实验比较：(1) decode阶段TPOT对比HBM4 baseline，RoMe在DeepSeek-V3/Grok 1/Llama 3上分别降低10.4%/10.2%/9.0%；(2) channel load balance rate (LBR)评估4KB粒度下的数据分布均匀性；(3) DRAM energy比较，RoMe分别降低1.9%/0.7%/0.7%；(4) MC scheduling logic面积对比，RoMe仅占conventional MC的9.1%。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  LLMSimulator（系统级LLM推理模拟器，支持continuous batching）+ Ramulator 2.0（cycle-accurate DRAM simulator，开源 https://github.com/CMU-SAFARI/ramulator2）。面积评估使用Synopsys Design Compiler + 7nm工艺。能量模型基于HBM4 energy model from prior work [2]。LLMSimulator论文来源为[77]，其开源状态论文未说明。RoMe的Ramulator 2.0修改代码状态论文未明确开源。

- 模拟器模拟什么的性能，修改了什么。
  模拟目标：单accelerator配置为8×HBM4 cubes、256GB memory、16TB/s bandwidth、4480 TFLOPS BF16、280 Op/B arithmetic intensity；多accelerator系统为8 accelerators，每accelerator 560 TFLOPS、256GB memory、16TB/s。模拟输入：LLM模型规格（DeepSeek-V3/Grok 1/Llama 3-405B）+ batch size (8-1024) + sequence length (8K)。模拟输出：TPOT、DRAM energy、channel LBR。论文修改：(1) Ramulator 2.0中实现RoMe memory system——将MC和DRAM配置为处理4KB request而非32B，实现VBA架构、command generator in logic die、simplified MC state machine（4 states），配置36 channels/cube（vs baseline 32）；(2) address mapping sweep确保baseline和RoMe都最大化bandwidth utilization；(3) baseline MC使用FR-FCFS scheduling + open-page policy + per-bank refresh；RoMe MC使用VBA interleaving + oldest-first + per-bank refresh with 2×tREFIpb interval。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：RoMe Ramulator 2.0修改和Verilog实现论文未明确开源。Ramulator 2.0开源 (https://github.com/CMU-SAFARI/ramulator2)。LLMSimulator链接论文未提供。使用流程（基于论文描述）：
  1. 配置输入：accelerator配置（8 cubes, BF16 throughput, arithmetic intensity, HBM4 timing parameters）+ RoMe配置（36 channels/cube, 4KB row size, VBA organization, command generator timing）+ workload（LLM model + batch size + seq len + parallelism strategy）
  2. 系统模拟：LLMSimulator建模LLM computation→生成memory request trace→Ramulator 2.0 cycle-accurate simulation of DRAM→RoMe MC processes 4KB requests with simplified scheduler→command generator in logic die expands RD_row/WR_row to ACT/RD/PRE sequence→VBA interleaving across two banks per VBA→data delivery。
  3. 输出：decode阶段TPOT (ms/token)，prefill阶段execution time，channel LBR，DRAM energy breakdown (ACT/RD/WR/PRE/interposer/command generator)。
  4. 面积验证：command generator和MC scheduler以Verilog实现→Synopsys Design Compiler 7nm综合→command generator area 4268.8µm² (0.003% logic die)；RoMe MC scheduling logic为conventional MC的9.1%。
