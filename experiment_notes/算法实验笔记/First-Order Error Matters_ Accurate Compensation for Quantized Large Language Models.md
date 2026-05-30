## First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 FOEM（First-Order Enhanced Method），一种改进 GPTQ 的 PTQ 方法。核心创新是在量化误差补偿中显式引入一阶梯度项。GPTQ 仅用二阶泰勒展开（假设一阶项为零），但 FOEM 发现逐列量化过程中，对已量化列的补偿会导致剩余未量化 latent weights 偏离 full-precision 权重，产生不可忽略的一阶梯度。FOEM 通过 g(W) ≈ β(W − 𝕎)H 近似梯度（无需反向传播），代入 Lagrangian 求解后 Hessian 项自动消去，仅增加轻量权重差分运算。实验对比：
  - Baseline：FP16（全精度上限）、RTN（round-to-nearest）、GPTQ、GPTAQ
  - 量化配置：weight-only（W4A16、W3A16，group size 128），weight-activation（W4A4KV4 结合 SpinQuant 预训练旋转矩阵）
  - 评估指标：WikiText2/C4 perplexity（PPL），PIQA/ARC-Easy/ARC-Challenge/HellaSwag/Winogrande/BoolQ 零样本准确率，5-shot MMLU
  - 消融：β ∈ {0.1, 0.2, ..., 1.0} 灵敏度分析（W3A16 Llama3-8B），β≤0.5 持续提升，β>0.5 急剧退化
  - 架构泛化：SSM 模型 Mamba-1.4B（W3A16）

- **硬件平台是什么，配置是什么。**
  量化校准：单卡 NVIDIA A800-80GB GPU。70B 模型评估需 2× A800 GPU。推理速度测试使用 vLLM 部署。校准数据：C4 数据集随机 128 条序列，序列长度 2048。β=0.1（所有实验默认）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：Llama2-7B/13B、Llama3-8B/70B、Llama3.2-1B/3B、Qwen3-8B、Phi-1.5B、Mistral-7B、Mamba-1.4B
  - 校准数据：C4（128 samples, seq_len=2048）
  - Perplexity：WikiText2、C4
  - 零样本常识推理：PIQA、Winogrande、ARC-Easy、ARC-Challenge、HellaSwag、BoolQ
  - 知识推理：5-shot MMLU

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Xingyu-Zheng/FOEM（AAAI 2026）。基于 gptqmodel 库集成。

  **算法pipeline核心流程（Algorithm 1：FOEM 单层量化）**：

  输入：FP 权重 W（m×n），校准输入 X，block size B
  1. 计算 Hessian：H = XX^T
  2. Cholesky 分解：L = Inverse_Cholesky(H + λI)，T = L^T（上三角）
  3. 保存原始权重副本：𝕎 ← W
  4. 按 block（size B）迭代：
     a. 对 block 内每列 j（j=i,...,i+B-1）：
        - 量化：Q_{:,j} ← quant(W_{:,j})
        - 一阶增强误差：E_{:,j-i} ← ((W_{:,j} − Q_{:,j}) − β(W_{:,j} − 𝕎_{:,j})) / T_{jj}
        - 补偿当前 block 内后续列：W_{:,j:(i+B)} ← W_{:,j:(i+B)} − E_{:,j-i} · T_{j,j:(i+B)}^T − β(W_{:,j} − 𝕎_{:,j})
     b. 补偿 block 外后续列：W_{:,(i+B):} ← W_{:,(i+B):} − E · T_{i:(i+B),(i+B):}^T

  **与 GPTQ 的核心差异**：
  - GPTQ 的补偿项：δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}（仅二阶）
  - FOEM 的补偿项：δw = −((w_q − ŵ_q) − β(w_q − 𝕎e_q^T))/T_{qq} · T_{q,q:} − β(W − 𝕎)
    额外减去 β(W_{:,j} − 𝕎_{:,j}) 项和分子中的 β(w_q − 𝕎e_q^T) 项，来自一阶梯度的近似
  - 梯度近似原理：g(W) ≈ (W − 𝕎)H，代入 Lagrangian 解后 H 和 H^{−1} 自动消去，无需显式计算
  - 额外开销：仅权重差分运算，无矩阵乘法。Llama3-8B 量化时间：GPTQ 825.50s，FOEM 828.90s（仅 +0.4%）
