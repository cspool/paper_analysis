## S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning

- baseline方法是什么？
  - Baseline 方法有三类：(1) **LoRA**：在每个 transformer 层的权重矩阵旁插入低秩 adapter $x' = B \cdot A \cdot x$（$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$），参数效率高（仅 $2dr$ 可训练参数），但模型容量受限于扁平的单层低秩结构，无法根据 token 特性动态调整计算路径。(2) **MixLoRA (MoLRE)**：将多个 LoRA expert 组合为 flat MoE 层 $x' = \sum_{i=1}^s \text{ROUTE}(x)^i \cdot B^i \cdot A^i \cdot x$，通过 top-k 路由为不同 token 激活不同 expert 组合。虽然增加了容量，但 expert 之间缺乏结构关系，路由灵活性仅来自"选择哪些 expert"，无法利用 expert 之间的连接方式产生额外的表达能力。(3) **HydraLoRA**：将 LoRA 的 up-projection 矩阵 B 拆分为多个 head，通过 dense gate 加权组合多 head 输出。类似 MoE 变体，但仍是单层结构，且参数利用率低（增加参数不提升准确率）。
  - 全栈执行例子（Baseline: MixLoRA on LLaMA 3-8B, A100 GPU, single token）：
    - 算法层：token embedding x 输入 flat MoE → 路由器对 s=8 个 expert 打分 → 选择 top-2 → 计算 $x' = \alpha_1 \cdot B^1 A^1 x + \alpha_2 \cdot B^2 A^2 x$ → 加回 frozen pre-trained 输出。不同 token 可激活不同 expert 对，但所有 8 选 2 组合 = $\binom{8}{2}=28$ 种可能输出。
    - 系统框架层：基于 HuggingFace PEFT + LLaMA-Factory SFT pipeline 训练，PyTorch 原生实现。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。
    - 硬件架构层：4× NVIDIA A100 80GB GPU，标准 PyTorch CUDA kernel 执行矩阵乘法。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**S'MoRE** 通过三个核心设计解决 baseline 缺陷：
    (1) **层次化结构式专家混合** 解决 "expert 缺乏结构关系导致路由灵活性受限" 的缺陷：将 expert 权重分解为多阶低秩残差 $W^i \approx \sum_{\ell=0}^{L-1} B^i_\ell A^i_\ell$，排列为 L 层结构。不同层的残差通过层间投影矩阵 $W_\ell$ 和 skip connection（原始 token x 直连每层残差）互联。同一组 activated expert 可以形成指数级多种非同构树结构，每种产生不同输出，从"选择哪些 expert"扩展为"expert 如何连接"。
    (2) **树形条件路由** 解决 "flat MoE 路由仅做 flat 选择" 的缺陷：路由器逐层自顶向下选择子节点，每层路由条件于祖先路径 $p(i_{\ell-1} \mid i_{L-1}, \dots, i_\ell, x)$，为每个 token 定制深度为 L 的专属激活树。路由概率通过 learnable key-query dot product + MLP 计算，支持 dense/sparse noisy top-k/switch 三种 gate 类型。
    (3) **非线性激活保证结构区分能力** 解决 "无结构关系时不同连接方式产生相同输出" 的缺陷：在每层聚合公式中引入非线性 $\sigma$（如 ReLU），使 S'MoRE 的 L 层传播模拟 Graph Isomorphism Network (GIN) 的 L 轮 Weisfeiler-Lehman (WL) test，理论上保证所有非同构树产生不同输出（Theorem 3.4）。无 $\sigma$ 时退化为 MoMOR，无法区分 Fig.3 中的非同构图。理论证明 S'MoRE 的 structural flexibility $\Gamma_{\text{S'MoRE}} = \prod_{\ell=0}^{L-1} \binom{s_\ell}{f_\ell}^{F_{\ell+1}}$ 比 MoMOR 上界 $\Gamma_{\text{MoMOR}}$ 呈指数级增长（Fig.2）。
  - 全栈执行例子（Method: 2-layer S'MoRE on LLaMA 3-8B, A100 GPU, single token）：
    - 算法层：token x 输入 → $x_{\text{down}} = W_{\text{down}} x$（降至 24 维）→ Layer 2 router: MLP₂ 生成 query，与 4 个 key vector 点积得 softmax score → 选 top-2 顶层 expert → Layer 1 router: 对每个父 expert，MLP₁(concat($x_{\text{down}}$, 父 key)) 生成 query → 选 top-2 子 expert → 形成激活树（2 个父节点各含 2 个子节点，共 4 条路径）→ 自底向上聚合：Layer 1 计算 $x_1^{p} = \sum_{n} \alpha_0^{p,n} \cdot \text{ReLU}(B_0^n A_0^n x)$ → Layer 2 计算 $x_2 = \sum_i \sum_n \alpha_1^{i,n} \cdot \text{ReLU}(B_1^n A_1^n x + W_1 x_1^{i \to n})$ → 最终投影 $x' = W_{\text{proj}} x_2$。不同 token 可获得 $\binom{4}{2} \times \binom{4}{2}^2 = 6 \times 6^2 = 216$ 种结构不同的激活树，远超 MixLoRA 的 $\binom{8}{2}=28$ 种。
    - 系统框架层：基于 HuggingFace PEFT 自定义 adapter 实现，LLaMA-Factory SFT pipeline 训练，OpenCompass 评估，PyTorch 原生实现。
    - 编译框架层：论文未明确说明。未来可集成到 vLLM 或 LMDeploy 等推理框架。
    - kernel调度层：论文未明确说明。论文提及可通过 CUDA kernel fusion 合并多层操作减少 kernel launch 开销，token-level parallelism（Triton kernel 或 torch.compile）交错不同层处理提升 GPU 利用率。
    - 硬件架构层：4× NVIDIA A100 80GB GPU。训练 wall-clock time 仅比 MixLoRA 增加约 24%（平均），router 计算代价相对 expert 传播最多 26%。
