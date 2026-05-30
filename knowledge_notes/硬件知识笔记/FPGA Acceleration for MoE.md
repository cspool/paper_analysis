## FPGA Acceleration for MoE

术语解释
利用FPGA（现场可编程门阵列）的可重构特性为MoE推理提供定制化硬件加速，通过设计专用数据路径和处理单元来优化MoE特有的稀疏计算和expert切换模式。

术语是什么？
FPGA加速MoE的关键优势：
- 可重构逻辑适应MoE的动态expert激活模式
- 定制数据路径针对稀疏计算优化
- 相比GPU能效比更高（适合边缘/嵌入场景）

三个代表性FPGA MoE加速方案：
- **FLAME**：首个全面利用MoE稀疏性的FPGA加速框架
  - 参数级：M:N剪枝减少不必要计算（平衡列均衡结构化剪枝与非结构化剪枝）
  - Expert级：CEPR（Circular Expert Prediction）改变expert激活路径模式提高预测准确率
  - 系统级：双缓冲机制在计算前一个expert的同时加载预测的下一个expert
  
- **M3ViT**：面向多任务ViT的FPGA架构
  - 仅激活与当前任务相关的稀疏expert通路
  - 硬件级协同设计实现任务间零开销切换
  
- **Edge-MoE**：首个端到端FPGA多任务ViT实现
  - GELU函数近似计算（降低FPGA上的实现复杂度）
  - 统一线性层模块实现硬件资源高效复用

从硬件架构角度拆解术语。
FLAME在FPGA上的双缓冲expert加载与计算流水线：
```
时间线（FPGA内部）：
Buffer 0: [Load Expert A] [Wait]       [Compute C]  [Load Expert E] ...
Buffer 1: [Idle]          [Compute A]  [Load D]     [Compute E]    ...
CEPR预测: [Predict B,C]   [Predict D]  [Predict E,F] ...

每个时钟周期：一个Buffer在计算，另一个Buffer在加载
效果：隐藏expert加载延迟，提高FPGA DSP/查找表利用率
```

术语一般如何实现？如何使用？
- 开发工具：Xilinx Vivado/Vitis HLS（高层次综合）
- 关键设计：expert权重存储在FPGA片外DRAM，通过DMA加载到片上BRAM
- M:N剪枝产生规则稀疏模式，适配FPGA的DSP阵列
- 双缓冲需要双倍的片上存储但实现计算与加载完全重叠
- 适用于嵌入式视觉（ViT MoE）、边缘NLP推理等低功耗场景

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models

---
