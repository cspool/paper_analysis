## BiLLM Pushing the Limit of Post-Training Quantization for LLMs

- baseline方法是什么？
  Baseline 是现有的 LLM PTQ 方法，包括 **GPTQ**（基于 Hessian 的二阶误差补偿 block-wise 量化，在 4-bit 表现良好但在 ≤2-bit 崩溃）、**PB-LLM**（部分二值化，保留 10% 权重为 INT8 其余二值化，平均 1.7-bit）、**RTN**（直接 round-to-nearest 量化，≤2-bit 完全崩溃）。

  Baseline 全栈执行例子（LLaMA-7B, GPTQ-2bit, block size=128）：
  - 算法pipeline：对每个 Linear 层的权重矩阵 W ∈ R^{n×m}，按 block size=128 逐列进行 2-bit 均匀量化（4 个量化级别），通过 Hessian 矩阵 H=2XX^T 进行 block-wise 误差补偿，即量化当前 block 后将误差 E = (W_q - W)/H^c 乘以 H^c 的对应子矩阵补偿到后续 block。PB-LLM 则先按 Hessian 选择 top-10% salient 元素保留 INT8，其余二值化，最终平均 1.7-bit。
  - 系统框架：PyTorch + HuggingFace Transformers 加载预训练模型，在单卡 NVIDIA A100 80GB 上完成 PTQ 过程。Calibration data 为 C4 的 128×2048 tokens。
  - 编译框架：论文未明确说明。模型量化后以 PyTorch 自定义 Linear 层加载，推理时在 Python 层进行反量化。
  - kernel调度：论文未明确说明。量化权重存储为 packed integer 格式，推理时反量化为 FP16 计算。
  - 硬件架构：NVIDIA A100 80GB GPU（Ampere 架构），无自定义硬件加速器。

  **Baseline 的核心缺陷：**
  1. GPTQ 在 ≤2-bit 时，均匀量化只有 4 个离散值（2-bit），无法表达 LLM 权重的钟形分布和少数 salient 权重的极端值，导致性能崩溃（LLaMA-7B 2-bit PPL=152.31 vs FP16=5.68）。
  2. PB-LLM 虽然保留 10% salient 权重为 INT8，但（a）非结构化选择需要 1-bit bitmap index 导致额外存储开销；（b）简单二值化 salients 的量化误差仍然很大；（c）未处理非 salient 权重的非均匀钟形分布，二值化误差随分布非均匀性增加。
  3. Vanilla RTN 在 ≤2-bit 完全崩溃（如 OPT-6.7B 2-bit PPL=28363.14），因无任何误差补偿机制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BiLLM 通过三项创新设计解决上述缺陷：

  **(1) 结构化 Salient 列选择 + 二进制残差逼近**（对应解决 PB-LLM 缺陷 a, b）：
  观察到 LLM 的 Hessian salient 权重在特定列中聚集（尤其在 Q/K/V 的 attention 投影层和 Out Projection 层），因此采用**按列结构化选择**而非逐元素非结构化选择。通过搜索最优 salient 列数 n* 最小化整体二值化重构误差。对选中的 salient 列，不保留 INT8（浪费位宽），也不简单二值化（误差大），而是使用**二进制残差逼近**：先二值化得到 B_o，再对残差 (W - α_o·B_o) 进行第二次二值化得到 B_r，最终用 α_o·B_o + α_r·B_r 两个二值矩阵之和逼近原始 salient 权重。这相当于用 2-bit 表达 salient 权重（vs PB-LLM 的 8-bit），且可证明残差逼近的量化误差 ε_rb ≤ 直接二值化的 ε_direct。

  **(2) 钟形分布最优分裂二值化**（对应解决 baseline 缺陷 c）：
  观察到非 salient 权重呈钟形分布（类似高斯/拉普拉斯），二值化作为极端均匀量化在此分布上误差极大。BiLLM 搜索一个最优分裂点 p*，将分布分为稀疏区（|w|>p，远离 0 的值）和集中区（|w|≤p，聚集在 0 附近），分别以独立的 scaling factor α_s 和 α_c 进行二值化。这相当于在钟形分布上用两个分段常数逼近，显著降低二值化 MSQE。搜索策略使用百分位搜索（步长 0.1），目标 min_p θ²_q,p = ||W_s - α_s·B_s||² + ||W_c - α_c·B_c||²。尽管实际分布偏离理想高斯，搜索曲线仍呈凸性，保证可找到最优 p*。

  **(3) Block-wise OBC 误差补偿**（继承自 GPTQ，Block size=128）：
  移除 column-wise 补偿以提升 PTQ 效率，仅保留 block-wise 补偿，确保分布探索不受干扰。

  **BiLLM 全栈执行例子（LLaMA-7B, ~1.09-bit）：**
  - 算法pipeline：加载 LLaMA-7B FP16 权重 → 提取 C4 calibration data（128×2048 tokens）→ 计算每层 Hessian H=2XX^T → Cholesky 稳定求逆 → 对每个 Linear 层逐 block（128 列）处理：① 计算 S = W²/H² 逐元素显著矩阵 → 按列聚合显著性 → 搜索最优 salient 列数（3-30 列）→ ② salient 列：residual binarization (B_o + B_r, 2-bit) → ③ 非 salient 列：搜索最优 p* (百分位 0.1-0.9) → split binarization (1-bit + 1-bit flag) → ④ 合并 + OBC block-wise 补偿 → 存储为 packed binary + scaling factors + bitmap。推理时：加载 packed binary → 按列解包 scaling factors → 需要时反量化 → FP16 GEMM，但论文主要关注 memory footprint 而非推理加速。
  - 系统框架：PyTorch + HuggingFace Transformers。自定义 quantized Linear 层替代原始 FP16 Linear。量化后模型以自定义 format 存储（packed binary + scaling factors + group/sparse-concentrated bitmap）。推理时可通过 custom kernel 实现 memory-efficient 推理，论文未提供完整推理 engine。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。量化权重以 1/2-bit packed binary 格式存储，推理时需反量化回 FP16 进行 GEMM。论文指出 "binarized GEMM is hard to implement directly due to fine-grained grouping"，主要收益在于 GPU memory footprint 降低（而非 GEMM 加速）。
  - 硬件架构：NVIDIA A100 80GB GPU。无自定义硬件加速器。

  关键设计动机映射：
  - GPTQ/PB-LLM 的非结构化 salient 选择浪费 bitmap 存储 + INT8 保留位宽过高 → BiLLM 的结构化列选择（利用 attention 层中 sensitivity 的列聚集特性）+ 残差二值化（2-bit 替代 8-bit 表达 salient 权重）
  - 钟形分布下直接二值化 MSQE 极大（因权重非均匀分布）→ optimal splitting 将分布分为 concentrate/sparse 两区独立二值化，用分段常数逼近降低误差
  - 简单二值化误差过高导致 ≤2-bit 崩溃 → 残差逼近可证明降低误差（ε_rb ≤ ε_direct）+ splitting 搜索凸性保证最优解
  - PB-LLM 平均 1.7-bit 仍过高（30%+ INT8 权重）→ BiLLM 将平均 bit-width 推至 1.07-1.13 bit（接近理论下限），且 PPL 更低
