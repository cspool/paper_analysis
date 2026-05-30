## UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  UniQL实现了一个**融合RoPE kernel**（fused rotary positional embedding kernel），以及**INT4在线解包-剪枝-重打包**的运行时kernel。

  融合RoPE kernel：结构化排序破坏了RoPE的原始位置嵌入索引（因Q/K权重列的对称排序使通道顺序改变）。为最小化内存访问，UniQL将gathering和slicing rotary positional embeddings的索引收集操作融合到单个kernel中。排序时采用对称排序策略：将norm score向量s ∈ R^{D_hd}对半分为[s1, s2]，对s1+s2排序得到对称索引向量idx_sym = [argsort(s1+s2), D_hd/2 + argsort(s1+s2)]，只需存储一半索引（硬件高效）。RoPE kernel直接使用该对称索引从旋转位置嵌入表中gather对应位置，避免多次内存往返。

  INT4在线处理kernel：设备端部署4-bit量化模型后，运行时的INT4权重需要：在线解包（unpack from INT4 to computation format）→ 按指定剪枝率去除末尾通道 → 重新打包为INT32向量 → 送入矩阵乘法kernel。

  实验比较：有/无融合RoPE kernel的延迟对比（Table 9），以及UniQL vs TRT-AWQ和TAO-HQQ在A6000和Jetson Orin Nano 8G上的TPOT和TTLT延迟。

- 后端平台是什么，配置是什么。
  NVIDIA A6000 GPU（48GB显存）：云端推理，测量TPOT（time-per-output-token）和TTLT（time-to-last-token），配置1024 prefill + 1024 generation tokens。
  NVIDIA Jetson Orin Nano 8GB：边缘设备推理，统一内存架构，测量TPOT和TTLT，配置512 prefill + 512 generation tokens。
  在Nano 8G上FP16模型OOM无法运行，必须使用量化模型。

- 评估性能的软件/脚本是什么。修改了什么。
  Kernel基础实现改编自：Marlin 4-bit kernels（Frantar et al., 2024）和Liger-Kernel RoPE kernels（Hsu et al., 2025）。
  
  修改内容：
  1. **融合RoPE kernel**：在原有RoPE实现中加入索引gather逻辑。传统做法是先将排序后的索引向量传入，再从sin/cos表中分别取出对应位置再应用旋转——这需要多次global memory访问。融合kernel在单个CUDA kernel中完成gather + slice + RoPE旋转计算，减少10%延迟（1.1× speedup for 4-bit Llama-3.1-8B at 0%和25%剪枝，Table 9）。
  2. **INT4运行时处理**：在Marlin 4-bit kernel中增加在线通道剪枝功能——解包INT4权重后，按剪枝率去除末尾通道，重打包为INT32供后续矩阵乘法使用。这允许同一量化模型在不同设备负载下支持0%-35%可变剪枝率。

  延迟profiling：每配置运行20次测量（5次warmup后），报告平均TPOT和TTLT。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/enyac-group/UniQL
  
  Kernel执行流程（以Llama-3.1-8B的Q/K投影后RoPE为例）：
  ```
  # 输入：已排序的WQ' = WQ @ S_qk, WK' = WK @ S_qk
  # 对称排序索引：idx_sym ∈ R^{D_hd}, D_hd=128 for Llama-3.1-8B

  # 传统两阶段（无融合）：
  X_q = X_h @ WQ'                              # [T, D_hd]
  cos_gathered = cos_table[idx_sym]             # 从cos表中gather
  sin_gathered = sin_table[idx_sym]             # 从sin表中gather
  X_q_rope = cos_gathered ⊙ X_q + sin_gathered ⊙ rotate_half(X_q)
  # 以上需要3次global memory往返

  # UniQL融合kernel（单kernel完成）：
  # 在同一个thread block中：
  For each position t:
      For each half-dimension pair (2d, 2d+1):
          i = idx_sym[d]                       # 从寄存器中的对称索引
          cos_val = cos_table[t, i]            # fused gather
          sin_val = sin_table[t, i]
          x0 = X_q[t, 2d]; x1 = X_q[t, 2d+1]
          X_q_rope[t, 2d]   = cos_val * x0 - sin_val * x1
          X_q_rope[t, 2d+1] = cos_val * x1 + sin_val * x0
  ```

  评估原理（延迟profiling）：
  - **TPOT（Time Per Output Token）**：测量decode阶段每个生成token的平均耗时。用CUDA events记录每个decode iteration的开始和结束，取20次运行平均。
  - **TTLT（Time To Last Token）**：从prefill开始到最后一个token生成的端到端总耗时 = prefill时间 + TPOT × 生成token数。
  - **Orin Nano 8G上的完整推理流程**：加载4-bit UniQL模型 → 设备端按当前负载配置剪枝率（0%-35%）→ 在线解包INT4权重 → 去除末尾通道 → 重打包 → Run inference。W4A16模型在Nano上TPOT从TAO-HQQ的133.6ms降至77.2ms（Qwen-2.5-7B），35%剪枝进一步降至57.7ms（2.3× vs TAO-HQQ）。
