## Attention Buffer / 片上注意力缓冲区（On-Chip KV Cache）

术语是什么？
Attention Buffer是HNLPU芯片内专用于存储KV Cache的SRAM模块。每个芯片配备320MB Attention Buffer，由20,000个bank组成（每bank 16KB），采用1W1R（一写一读）端口配置和32-bit访问宽度。该Buffer在worst-case PVT条件下维持80 TB/s带宽和3-cycle延迟。主要功能是作为芯片所分配attention groups的KV Cache，仅在on-chip容量不足时才将多余KV条目卸载到HBM。

从硬件架构角度拆解：
Attention Buffer在HNLPU硬件中的运转：(1) Decode阶段，VEX单元从Attention Buffer读取K/V pairs用于attention score计算（每芯片每周期处理32 cached KV-heads无停顿）；(2) 新生成的K/V pairs写入Buffer，与读取并发（利用1W1R双端口）；(3) 当context length增大使on-chip容量溢出时，额外KV条目通过HBM PHY写入off-chip HBM，后续访问时加载回Buffer；(4) Buffer同时存储FFN block中residual connections的activation向量。论文的execution time分析显示，双缓冲机制有效隐藏了memory access延迟——stalls在最高256K tokens时仍可忽略，仅在512K extreme context length时达到10.7%。

术语一般如何实现？如何使用？
Memory Compiler在5nm工艺节点生成。Attention Buffer是HNLPU功耗的第二大来源（27.8%，85.73W），但通过将KV Cache保持在on-chip避免了GPU方案中HBM访问的高功耗。结合HBM（8-stack，24GB/stack，$10-20/GB）形成两级KV Cache层次：on-chip SRAM（低延迟高带宽）→off-chip HBM（高容量）。

涉及论文标题：
- Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

