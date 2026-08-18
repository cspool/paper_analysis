## 扩展 BitMoD cycle-accurate 模拟器（post-synthesis timing/energy 模型注入的加速器系统级评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 UNICORE 论文的加速器系统级评估方法论：以 BitMoD（HPCA 2025，Bit-serial Mixture-of-Datatype LLM Acceleration）的 cycle-accurate 模拟器为基础，把 UNICORE post-synthesis RTL 导出的 timing 与 energy 模型注入模拟器，评估端到端 prefill/decode 性能与能耗。片上 SRAM（512KB activation buffer + 512KB weight buffer）用 CACTI 建模，片外访存能耗用 DRAMsim3（DDR4 25.6 GB/s 用于 prefill，decode 额外加 HBM2 256 GB/s）建模。全部加速器（UNICORE 与五个 baseline）在相同面积约束、相同 buffer 配置、相同 TSMC 28nm/1GHz 综合流程下比较。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 运转流程：① RTL（SpinalHDL）经 Synopsys Design Compiler 在 TSMC 28nm/1GHz 综合，导出各硬件块（S-FPMA slice 阵列、双链累加、格式转换/补偿管线、Rescale）的面积/时序/翻转活动；② 把这些 timing/energy 模型注入 BitMoD cycle-accurate 模拟器；③ 模拟器以 cycle 推进 prefill（序列长 8192、Llama-2-7B/Llama-3-8B）与 decode（batch=128）执行，权重/激活按 Group 流经格式转换→PreAdd→PE 阵列→双链累加→Rescale→全局累加，片外访存由 DRAMsim3 计时计能耗、片上 buffer 由 CACTI 计面积能耗；④ 输出归一化加速比（图 17/19/20）、能耗分解与 TOPS/W（图 18）。baseline（OliVe/Tender/M-ANT/BitMoD/AxCore）在同一框架下归一化实现（面积约束与 buffer 配置相同），Tender 扩展支持 W16A16、M-ANT/BitMoD 扩展整数激活量化以对齐评估协议。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：基于 BitMoD 开源模拟器（https://github.com/abdelfattah-lab/BitMoD-HPCA-25，Zenodo https://zenodo.org/records/14251941）扩展；UNICORE 模拟器在 artifact Software/Simulator/（Conda 环境 + 运行脚本输出 figure/*.pdf，复现图 17-20）。CACTI（https://github.com/HewlettPackard/cacti）与 DRAMsim3（https://github.com/UMD-MEMSYS/DRAMsim3）为通用开源建模工具。作用：在无真实硅片条件下把"位宽可扩展加法型数据通路 vs 乘法器型 O(n²) 塌缩"量化为面积归一化后的吞吐/加速比/能耗，验证同一硬件在 W4A4/W4A8/W8A8/W16A16 下保持高利用率。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
