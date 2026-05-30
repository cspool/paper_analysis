## DiJiang: Efficient Large Language Models through Compact Kernelization

- baseline方法是什么？
  Vanilla Transformer的自注意力机制：Attention(Q,K,V) = softmax(QK^T)V，计算复杂度O(n²d)，其中n为token数量、d为head维度。对于长序列，二次复杂度导致训练和推理成本急剧增长。以LLaMA2-7B为例，完整训练需约82,432 GPU-hours和~36 MWh电力。虽然Performer等方法通过Positive Random Features (PRF)可实现线性注意力，但其Monte Carlo采样近似效率仅为O(1/m^{-0.5})，需要m >> d才能保持性能，极大削弱了线性注意力带来的加速收益。其他线性Transformer方法（Linformer, Cosformer, RetNet）在fine-tuning场景下精度损失严重（Pythia-410M fine-tuning中最高仅Performer的0.4183 vs 原始0.454）。

  全栈执行例子（一条token序列通过vanilla Transformer推理）：
  - 算法Pipeline：对于每个token i，计算 o_i = Σ_j exp(q_i·k_j^T)/Z * v_j，其中对每对(i,j)做d维内积 → Softmax归一化 → 加权求和，O(n²d)复杂度。
  - Serving调度：论文未明确说明（依赖标准推理框架如HuggingFace Transformers/PyTorch）。
  - 编译框架：论文未明确说明（标准PyTorch eager/graph模式）。
  - Kernel调度：标准cuBLAS/cuDNN GEMM和Softmax kernel，未做针对性优化。
  - 硬件架构：NVIDIA A800 GPU，标准CUDA core和Tensor Core执行矩阵乘法。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DiJiang使用Frequency Domain Kernelization (FKA)将vanilla Transformer转换为线性复杂度模型，通过三个递进创新解决baseline痛点：

  **(1) Quasi-Monte Carlo (QMC)替代Monte Carlo**：传统PRF用Monte Carlo采样近似Gaussian核 e^{qk^T}，收敛速度仅O(1/m^{-0.5})。DiJiang基于Bochner定理将核函数转为球面积分，用渐近均匀点集（QMC）替代随机采样，收敛速度提升至O(1/m)，使m=d即可保持近似精度。
  
  **(2) 加权QMC (WPFF)**：进一步引入可学习权重D对标采样点进行加权，通过求解凸优化问题（最小化Paley-Wiener空间的discrepancy度量）获得最优权重配置，理论上证明WPFF的积分估计误差上界不大于PFF。
  
  **(3) DCT频域映射 (WDCF)**：将随机投影替换为确定性DCT系数矩阵C进行频域变换，利用DCT的快速算法（O(log m) vs O(m)）和能量集中特性（稀疏表示），在实数域操作无需处理复数，比FFT更高效且硬件友好。

  全栈执行例子（一条token序列通过DiJiang FKA推理）：
  - 算法Pipeline：给定Q,K,V ∈ R^{n×d}，先用DCT系数矩阵C计算频域映射 φ_WDCF(x) = D⊙exp(T·C·x^T)，再按 φ(Q)·φ(K)^T·V = φ(Q)×(φ(K)^T×V) 计算，先乘K^T×V得O(nmd)再乘φ(Q)。当m=d时总复杂度O(nd²)，序列长度n与计算量成线性关系（而非二次）。
  - Serving调度：论文未明确说明（继承预训练模型的推理管道）。
  - 编译框架：论文未明确说明。
  - Kernel调度：借鉴RetNet的高效推理实现，利用线性注意力可合并K和V计算的特点，实现O(1)每token推理开销；但论文未描述具体kernel实现细节。
  - 硬件架构：NVIDIA A800 GPU，利用DCT的快速算法（可通过FFT-like蝶形结构实现）在GPU上高效执行频域变换。

  **核心对比**：
  - 训练成本：DiJiang-7B仅需40B tokens（LLaMA2-7B用2000B tokens，约1/50）；DiJiang-410M训练6.6天 vs Pythia-410M 105.8天（约1/16）。
  - 推理速度：DiJiang-2.8B推理284 tokens/s vs Pythia-2.8B 34 tokens/s（约8.4×），且随token长度增加显存和延迟不增长（线性复杂度优势）。
  - 精度保持：DiJiang-410M平均0.4567 vs Pythia-410M 0.454（几乎无损）；DiJiang-7B平均0.557 vs LLaMA2-7B 0.565。
