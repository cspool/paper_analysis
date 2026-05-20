## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出ADAPT（Adaptive Dynamic Allocation for Parallel Tree），一种budget-constrained integer programming方法，实现hardware-aware dynamic draft generation。核心设计：(1) 问题形式化：给定computational budget B和draft model在各位置的概率分布，目标为最大化expected number of successfully verified tokens。决策变量x_{i,j,l}∈{0,1}表示是否在第i层第j个node选择第l个token做branching，目标函数max Σ_{i=1}^{D} Σ_{j=1}^{N_i} Σ_{l=1}^{V} p_{i,j,l}·x_{i,j,l}；(2) 三重约束：computational budget constraint（总branch数≤B）、structural constraint（每层branch数≤k_max，k_max设为FPGA parallel support number的整数倍如8/16/32）、pipeline depth constraint（D ≥ D_min = ⌈T_verify / T_draft⌉，确保heterogeneous pipeline的compute overlap最大化）；(3) Temperature-Controlled Probabilistic Sampling Greedy Approximation：NP-hard问题的实时求解——定义Path Cumulative Probability P_cum(i,j,l) = p_{i,j,l}·∏_{k} p_{k,par(a_k),a_k}，用softmax temperature T归一化后通过Gumbel sampling做非重复选择，T→0退化为deterministic top-k selection，T较大趋向uniform exploration；(4) 算法复杂度：O(D·k_max·log k_max) time，O(D·k_max) space。实验比较：(1) acceptance rate稳定在75%-85%（Qwen3-0.6B/8B pair）；(2) 动态draft length分布呈long-tail特征；(3) hyperparameter sensitivity：confidence threshold ε和temperature T对acceptance rate的影响（ε≤0.6时acceptance rate>75%，T≤1时>80%）；(4) HW-Branch ablation贡献（2.21× speedup from baseline）。

- 硬件平台是什么，配置是什么。
  主平台：Intel Xeon 4310 + RTX 4090 (2230MHz, 330 FP16 TOPS, 24GB) + V80 FPGA (300MHz, 10848 DSPs, HBM, 64GB)。附加：A100 + U200/V80 FPGA。算法参数：k_max设为FPGA并行支持数的整数倍（8/16/32），D_min由T_verify/T_draft ratio确定。

- 模型是什么。数据集和bench分别是什么。
  目标模型：Vicuna-7B-v1.3 (4096 hidden, 11008 FFN, 32 layers)、LLaMA-2-7B (4096 hidden, 11008 FFN, 32 layers)、OPT-13B (5120 hidden, 20480 FFN, 40 layers)、Qwen3-8B (4096 hidden, 12288 FFN, 36 layers)。Draft模型：Vicuna-160M (768 hidden, 3072 FFN, 12 layers)、LLaMA-160M (768 hidden, 3072 FFN, 12 layers)、OPT-125M (768 hidden, 3072 FFN, 12 layers)、Qwen3-0.6B (1024 hidden, 3072 FFN, 28 layers)。数据集（Spec-Bench）：MT-Bench (multi-turn对话)、Translation (WMT)、Summarization (CNN/DailyMail, XSum)、Question Answering (SQuAD, Natural Questions)、Math Reasoning (GSM8K, MATH)、Retrieval-Augmented Generation (RAG)。本文Fig.12/13使用Qwen3-0.6B/Qwen3-8B pair。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  已开源：https://github.com/ShaoqiangLu/DFVG（MIT License）。ADAPT算法pipeline：
  1. 每轮decoding iteration：FPGA draft model对当前prefix做forward pass，输出vocabulary-level probability distribution
  2. Budget allocation：根据B（FPGA hardware parallelism limit确定的max branch数）和D_min（T_verify/T_draft ratio）确定本iteration的depth和branching budget
  3. Tree construction：从root node起逐层执行temperature-controlled Gumbel sampling——每层计算各候选token的Path Cumulative Probability，用softmax+温度T归一化→Gumbel noise perturbation→argmax选择k_i个非重复候选→下一层将各selected token作为新node继续扩展
  4. 生成的token tree通过PCIe传输到GPU，TreeSort-Verify做block-parallel attention verification
  5. GPU返回accepted tokens→FPGA根据返回的accepted prefix length检测rollback（对比local sequence），若rollback则reset KV cache并从verified prefix恢复
  例如Qwen3-0.6B/8B pair下，acceptance rate 75%-85%，draft length呈long-tail分布（多数iteration只需短draft，少数需长draft），动态分配避免了static draft length的"too short→frequent rollback / too long→computational waste"困境。k_max设为FPGA PE array的整数倍（如16），确保每次生成的branch数对齐硬件并行粒度。
