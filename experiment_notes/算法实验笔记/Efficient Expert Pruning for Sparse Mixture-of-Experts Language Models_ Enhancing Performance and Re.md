## Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **EEP (Efficient Expert Pruning)**，一种无梯度进化策略（gradient-free evolutionary strategy），用于 SMoE 模型的 expert 剪枝和合并。EEP 分为两阶段：(1) **Expert Pruning Phase**：通过进化搜索找到最优的 expert 剪枝模式，引入 Router Mapping 矩阵 WRM ∈ ℝ^{E'×E} 和 Expert Merging 矩阵 WEM ∈ ℝ^{E'×E}，两矩阵初始化为 one-hot 行向量，仅保留选中的 expert 权重及对应路由权重。此阶段不更新任何网络参数(no gradient computation)。(2) **Expert Merging Phase**：WRM 和 WEM 解耦，从离散 0/1 值过渡到连续值，将剪枝掉的 expert 知识融合到保留的 expert 中，通过 block-wise weighted sum 合并 expert 权重: θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}。进化搜索采用 Crossover + Mutation + Selection 迭代优化。
  实验比较包含两大使用场景：(a) **减少 total expert 数量**（节省 GPU 显存）：EEP vs Random Selection / Frequency-based pruning / Soft Activation pruning / NAEE（对 Mixtral 8×7B 从 8 expert 剪枝到 4/2；Mixtral 8×22B 剪枝到 4/2；Qwen1.5-MoE 从 60→30/15；Qwen2-MoE 从 64→32/16/8/4/2/1）；(b) **减少 active expert 数量**（加速推理）：Top-2→Top-1 with EEP merging vs Full Model / NAEE Dynamic Skipping；(c) 组合场景：total=4, active=1；(d) 泛化测试：MMLU 50+7 split, IID + OOD；(e) 消融实验：group number (4 vs 32), search iterations；(f) 微调场景：EEP 用于不剪枝情况下的 fine-tuning；(g) 性能画像：显存占用和推理加速比。

- 硬件平台是什么，配置是什么。
  主要实验在 NVIDIA GPU 上完成（具体型号论文未明确说明计算卡型）。性能画像实验使用 2× NVIDIA A100 GPU，batch size=256 测试 SQuAD 数据集。EEP search 过程仅需推理（无梯度计算），可在仅支持推理的设备上运行。搜索过程：Pruning Phase 40 iterations + Expert Merging Phase 160 iterations。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mixtral 8×7B-Instruct (8 experts, top-2, 47B total/13B active)；(2) Mixtral 8×22B-Instruct (8 experts, top-2, 141B total/39B active)；(3) Qwen1.5-MoE-A2.7B-Chat (60 experts, top-4, 14.3B total/2.7B active)；(4) Qwen2-MoE-A14B-Chat (64 experts, top-4, 57B total/14B active)。
  数据集/benchmarks：(1) SuperGLUE tasks: COPA, MultiRC, WIC, WSC, RTE, BoolQ, CB, ReCoRD；(2) SQuAD（阅读理解和问答）、DROP（离散推理阅读理解）；(3) MMLU（57 数据集，用于 IID/OOD 泛化测试）。每个数据集随机抽取训练集子集做进化搜索，测试集做评估。所有数据集使用统一生成式评估方法，基于 OpenCompass 框架实现 prompt 设计和模板匹配。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自清华大学、Infinigence AI、KU Leuven、Microsoft Research、上海交通大学。代码开源在 https://github.com/imagination-research/EEP。

  **EEP 算法核心流程（对应论文 Algorithm 1）**：

  ```
  Algorithm: Evolutionary Search of EEP
  Input: 全量 expert 权重 Θ={θ^l_1,...,θ^l_E}_{l=1..L}, 评估器 F
  Hyperparams: Epochs (迭代轮数), M_CP (candidate parents 数量), Iter (每轮 mutation 数)
  Output: 最优 W* = {W^l_EM, W^l_RM}_{l=1..L}

  1: P ← ∅
  2: 随机初始化 W_init，确保每行是 one-hot vector
  3: P ← P ∪ {(W_init, F(W_init))}
  4: for phase in {Pruning Phase, Merging Phase}:
  5:   for t = 1..Iters:
  6:     NG ← ∅
  7:     for i = 1..Epochs:
  8:       CP ← {W_i | F(W_i·Θ) ranks within top min(M_CP,|P|) in P}
  9:       W_f, W_m ← RandomSample(CP)  // 从候选父代中随机采样两个
  10:      W_new ← Mutate(Crossover(W_f, W_m))
  11:      NG ← NG ∪ {(W_new, F(W_new))}
  12:     P ← P ∪ NG
  13: return W* ← argmin F(W)
  ```

  **Expert Pruning Phase 张量计算**：
  - WRM, WEM 初始化为 one-hot rows (每行仅一个元素为1，其余为0)
  - 且约束 WRM = WEM
  - Router 变换: G' = WRM · softmax(Z · W_G)，将 E 维路由权重降为 E' 维
  - Expert 剪枝: θ'_j = WEM 的 one-hot 行选择对应的原始 expert 权重

  **Expert Merging Phase 张量计算**：
  - WRM 和 WEM 解耦，元素从离散 0/1 过渡到连续值
  - 对第 j 个新 expert: θ'_j = {Σ_{i=1}^{E} ω_ji · W₁i, Σ_{i=1}^{E} ω_ji · W₂i, Σ_{i=1}^{E} ω_ji · W₃i}
  - 其中 ω_ji 来自 WEM 的第 j 行第 i 列
  - 可在不更新任何网络参数的情况下完成，仅需模型推理

  **Crossover 操作**：沿 retained expert 维度组合父代的 merging coefficients。
  **Mutation 操作**：Pruning Phase 随机替换 pruned experts 为其他 experts 并相应设置路由权重；Merging Phase 对 merging coefficients 逐元素加入 Gaussian noise。

  **关键结果**：Mixtral 8×7B 剪枝 75% experts（8→2），参数减少 72%，性能与 full model 可比。剪枝 50% experts 在 SQuAD 上准确率从 53.4% 提升至 75.4%。Active experts 从 2→1 实现 prefill 加速 1.63×。4 total + 1 active expert 组合节省 47% GPU 显存，1.41× 推理加速。
