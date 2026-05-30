## DiJiang: Efficient Large Language Models through Compact Kernelization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Frequency Domain Kernelization (FKA) —— 通过加权Quasi-Monte Carlo采样和DCT（离散余弦变换）将预训练vanilla Transformer的softmax注意力替换为线性复杂度的核化注意力，仅需少量fine-tuning即可将二次注意力O(n²d)降至线性O(nmd)（设m=d）。核心步骤：(1) 基于Bochner定理将Gaussian核（softmax注意力的等价形式）转为积分形式；(2) 用加权Quasi-Monte Carlo（PFF→WPFF）替代Monte Carlo采样以提升近似效率（O(1/m) vs O(1/m^{-0.5})）；(3) 用DCT系数矩阵C替换随机投影进行频域映射（WDCF），将复杂度从O(m)降至O(log m)。
  - 实验比较：(a) 不同模型规模（70M～2.8B Pythia）fine-tuning性能与训练时间；(b) 跨模型泛化（OPT-350M, TinyLLaMA-1.1B, LLaMA2-7B）；(c) 与Linformer/Performer/RetNet/Cosformer等线性注意力方法对比；(d) 推理吞吐与显存对比；(e) 注意力图可视化分析。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A800（训练时间测量及推理吞吐评估均使用A800）。
  - 推理评估token长度为2048。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Pythia（70M/160M/410M/1B/1.4B/2.8B系列，从HuggingFace EleutherAI checkpoint fine-tune）、OPT-350M、TinyLLaMA-1.1B、LLaMA2-7B。
  - 训练数据集：The Pile（825 GiB英语语料，22个子集）。
  - 评测benchmark：(小模型) PIQA, WinoGrande, WSC, ARC-E, ARC-C, LogiQA；(7B模型) 额外包含 SIQA, BoolQ, HellaSwag, MMLU, NQ, COPA, Race-Middle。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码已公开 https://github.com/YuchuanTian/DiJiang
  - 算法pipeline（Algorithm 1 + 核心公式）：

  **推理阶段——FKA前向计算（Equation 13）**：
  ```
  输入: Q, K, V ∈ R^{n×d}  (n=tokens, d=head_dim, 设定m=d)
  1. 构建 DCT 系数矩阵 C ∈ R^{m×d}:
     C[j1,j2] = s_j1 * s_j2 * Σ_i1 Σ_i2 cos(π(2i1+1)j1/(2d)) * cos(π(2i2+1)j2/(2d))
     其中 s_j = sqrt(1/d) 若 j=0, 否则 sqrt(2/d)
  2. 随机采样对角矩阵 T = diag(t_1,...,t_m), t_i ~ U(0,1) 服从逆累积分布
  3. 计算 WDCF 特征映射（Equation 12）:
     φ_WDCF(x) = D ⊙ exp(T · C · x^T)  对 x ∈ {q_i, k_i}
     其中 D ∈ R^m 为可学习权重
  4. 计算线性注意力（Equation 13）:
     FKA(Q,K,V) = φ_WDCF(Q) · φ_WDCF(K)^T · V
                 = φ(Q)_{n×m} × (φ(K)^T)_{m×n} × V_{n×d}
                 = φ(Q) × (φ(K)^T × V)    # 先算后两项 O(nmd)
                 → 输出 O ∈ R^{n×d}
  ```

  **训练/微调阶段（Algorithm 1）**：
  ```
  输入: 少量训练数据 x_i, 预训练Transformer模型 M
  1. 初始化每层的 DCT系数C, 权重D, 对角矩阵T
  2. 将每层的 Attention(Q,K,V)=softmax(QK^T)V 替换为 FKA(Q,K,V)=φ_WDCF(Q)·φ_WDCF(K)^T·V
  3. 得到变换后模型 M_FKA
  4. repeat:
       a. 从 x_i 随机采样mini-batch
       b. 用 M_FKA 前向传播
       c. 按loss和梯度更新 M_FKA 中的可学习参数
     until convergence
  输出: 高效语言模型 M_FKA
  ```

  - 关键设计：(a) 使用DCT替代FFT因为DCT在实数域操作，更少计算量且更硬件友好；(b) 设置m=d避免增加计算复杂度；(c) 借鉴RetNet的gating机制增强DiJiang；(d) 训练仅需原始Pythia约1/16的训练时间，DiJiang-7B仅需40B tokens训练（LLaMA2-7B用2T tokens）。

  **主要实验结果**：
  - DiJiang-410M vs Pythia-410M: 平均benchmark 0.456 vs 0.454，训练6.6天 vs 105.8天（~1/16），推理787 tokens/s vs 203 tokens/s（~3.9×）。
  - DiJiang-7B vs LLaMA2-7B: 平均benchmark 0.557 vs 0.565，训练数据40B tokens vs 2000B tokens（~1/50）。
  - DiJiang-2.8B vs Pythia-2.8B: 平均0.473 vs 0.478，训练37.1天 vs 593.3天（~1/16），推理284 tokens/s vs 34 tokens/s（~8.4×）。
  - 对比其他线性注意力（Pythia-410M fine-tuning）：DiJiang 0.4567（最佳），Performer 0.4183，Cosformer 0.4047，Linformer 0.3982，RetNet 0.3843。
