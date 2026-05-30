## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文是综述，不提供原始实验。它在硬件级别（Section 5）综述了以下kernel/运行时计算优化：
  - **FLAME**：首个在FPGA上全面利用MoE稀疏性的加速框架。参数级使用M:N剪枝减少不必要计算，expert级通过CEPR（Circular Expert Prediction）进行稀疏激活预测，使用双缓冲机制在计算前一个expert时加载预测的expert。
  - **M3ViT**：基于多任务场景中attention计算重排序的FPGA架构，只激活与当前任务相关的稀疏"expert"通路，实现任务间零开销切换。
  - **Edge-MoE**：首个端到端FPGA实现的多任务ViT，包括GELU函数近似计算、统一线性层模块实现硬件资源高效复用。
  - **MoE-CSP**：设计了处理4-bit/8-bit量化权重的专用CUDA kernel，执行浮点计算加速。
  - **QMoE**：实现了自定义压缩格式和定制GPU kernel用于1-bit on-the-fly计算。

- 后端平台是什么，配置是什么。
  - **FPGA平台**：Xilinx/Intel FPGA（FLAME、M3ViT、Edge-MoE使用的目标平台）
  - **GPU平台**：NVIDIA GPU（MoE-CSP、QMoE的CUDA kernel目标）
  - 论文未统一规定硬件配置

- 评估性能的软件/脚本是什么。修改了什么。
  - FLAME：修改了FPGA上的expert激活路径模式（circular expert prediction替代线性预测），实现了双缓冲加载机制
  - M3ViT：修改了attention计算顺序以支持多任务场景的稀疏expert激活
  - Edge-MoE：修改了GELU函数的FPGA实现（近似方法降低复杂度）和线性层模块（统一设计实现复用）
  - MoE-CSP：新增了处理4-bit/8-bit量化权重+浮点计算的CUDA kernel
  - QMoE：新增了1-bit压缩格式和相应的GPU反量化kernel

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源情况**：各硬件方法的开源情况各不相同，综述未统一收集各方法的开源链接。主要的开源加速框架（如DeepSpeed-MoE、Tutel）中有部分kernel优化开源。

  **FLAME FPGA Expert预测与双缓冲Kernel执行流程**：
  1. 输入：经过M:N剪枝后的expert权重矩阵W_pruned，当前token的expert激活历史
  2. CEPR预测：基于循环预测模式，改变expert激活路径的patterning，预测下一层所需expert集合E_pred
  3. 双缓冲加载：在计算当前expert E_curr的同时，通过第二个buffer预加载E_pred的权重
  4. Expert计算：FPGA上的DSP/查找表执行稀疏矩阵乘法
  5. 输出：当前token的expert输出y_i

  **MoE-CSP量化CUDA Kernel执行流程**：
  1. 输入：INT4/INT8量化的expert权重W_q，FP16的输入激活值x
  2. Kernel内反量化：w_deq = dequantize(W_q[i])，转换为FP16
  3. 浮点矩阵乘法：y_i = matmul(x, w_deq)
  4. 输出：FP16精度的expert输出
