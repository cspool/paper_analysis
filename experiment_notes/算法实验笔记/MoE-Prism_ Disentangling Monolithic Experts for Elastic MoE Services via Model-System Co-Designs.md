## MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Prism 的 Offline Refactoring Engine，通过以下三步将预训练 MoE 的 monolithic expert 分解为 fine-grained sub-expert，实现无重训练的模型弹性化：
    1. **Neuron Activation Profiler**：在 Wikitext-2-raw-v1 校准数据集上运行模型，从每个 expert 的 SwiGLU FFN 层捕捉中间激活矩阵 M（B×C，B=tokens, C=neurons），利用 FFN 每列计算独立性将"neuron"定义为负责 A 矩阵单列+对应 W_down 行的权重集合。
    2. **Partitioning Optimization Solver**：组合优化问题——寻找将 C 个 neuron 划分到 N 个子 expert 的最优分区 P*，目标是最小化所有 tokens 上被 deactivated sub-experts 的 L1 norm 之和。求解器分两阶段：Greedy Initialization（贪心负载均衡初始分区）+ Simulated Annealing Refinement（T0=100.0, α=0.995, I=100,000 次迭代）。
    3. **Gating Mechanism Reconstructor**：两种策略——(a) Training-Free Proxy Gating：构建 neuron 共激活矩阵 C_co = B^T·B，选择每个 sub-expert 内 centrality 最高的 r=4 个 neuron 作为 gate neurons，用其平均 L1 norm 估算 sub-expert 输出 norm；(b) Low-cost Router Finetuning：仅微调 router（<0.1% 参数），采用 curriculum training 逐步增加 k（8→24/32），在 SlimPajama 的 200K 序列上训练，LR=1e-5。
  - 实验比较：与原始 MoE 模型在同等激活参数量下的 Perplexity（Wikitext）和下游任务准确率（Winogrande 3-shot, ARC-C 5-shot, SciQ 0-shot, BoolQ 0-shot）对比。每个 expert 被划分为 N=4 个子 expert。
- 硬件平台是什么，配置是什么：NVIDIA H800 GPU（训练/评估），PyTorch 2.7.0 + CUDA 12.6。
- 模型是什么。数据集和bench分别是什么：模型为 OLMoE-1B-7B（7B, 64→256 experts, k=8→32）、DeepSeek-V2-Lite（16B, 64→256 experts, k=6→24, 含 2 shared experts）、Qwen3-30B-A3B（30B, 128→512 experts, k=8→32）。校准数据集为 Wikitext-2-raw-v1。微调数据集为 SlimPajama（200K 序列）。评估 benchmark 使用 lm-eval (Eleuther AI)，vLLM 作为推理后端。
- 开源情况：论文未明确说明开源链接。基于论文描述，算法流水线为：(1) 对预训练 MoE 在 Wikitext-2-raw-v1 上前向传播，从 FFN 中间层收集激活矩阵 M_e；(2) 对每个 expert，运行 SA 优化器求解最优分区 P*，输入为 M，输出为 N 个子 expert 的 neuron 索引映射；(3) 从 M 计算共激活矩阵 C_co = B^T·B，B 为 top-k_a 激活二值化矩阵；(4) 对每个子 expert S_n，选择 centrality 最高的 r 个 neuron 作为 gate neurons；(5) 可选：在 SlimPajama 上用 curriculum training 仅微调 linear gate（线性层），冻结其余所有参数；(6) 推理时，gate neurons 对其所属子 expert 的 L1 norm 做代理估算，router 按 softmax 分数选择 top-k 个子 expert，加权求和输出。
