## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoEBlaze 提出一种内存高效的 MoE 训练算法，核心创新有三点：
    1. **记忆高效的 token dispatch**：不创建传统的 per-expert materialized token buffer（大小为 L×K×d，在 DeepSeek 规模下约 94GB），而是生成四组轻量级索引数据结构——expert_token_indices（L×K）、expert_token_offsets（E+1）、token_expert_indices（L×K）、token_index_map（L×K）——通过 on-the-fly gather/scatter 直接从原始未重排激活张量进行 expert 计算和结果聚合。
    2. **前向传播**：Token dispatch 仅生成索引数据结构而不分配路由 token 显存；Expert 计算通过 per-expert token list 做 on-the-fly gather；Output aggregation 通过 per-token expert list 做 on-the-fly reduction。
    3. **反向传播**：利用相同的逆向映射索引，避免将 (L,d) 梯度"展开"为 (L×k,d) 路由梯度的中间步骤，通过 scatter 操作直接将输出梯度映射回对应位置。
  - 实验比较：(1) 训练速度（forward+backward 的 speedup vs MegaBlocks），(2) 激活内存消耗（PyTorch saved tensor hooks 追踪的中间激活张量总大小），在 SiLU 和 SwiGLU 两种激活函数下、7 种 MoE 配置（见表 1）下对比。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA H100 Tensor Core GPU（80GB HBM）。软件栈：PyTorch 2.0.1 + CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  - 模型：MoE 配置共 7 种（表 1）：input dim d ∈ {512, 1024, 2048}，expert 数 E ∈ {4, 8, 16}，top-k K ∈ {1, 2, 4}，batch size B ∈ {16, 32}，seq len L ∈ {512, 1024, 2048}。FFN hidden dim = 4×d。配置模拟常见 LLM 设定（如 DeepSeek 参数规模）。
  - 数据集/bench：论文使用这些配置的合成数据/随机张量进行单层 MoE 的 Sparse-to-Sparse 计算阶段评测，未使用具体 NLP benchmark。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源链接。HuggingFace papers 页面（https://huggingface.co/papers/2601.05296）无代码仓库链接。
  - 算法 pipeline 伪代码（核心前向）：
    1. 输入：x ∈ R^{L×d}，gate 权重 W_g ∈ R^{E×d}
    2. topk_experts = TopK(softmax(W_g · x))  // 形状 (L, k)
    3. 构建 expert_token_indices[L×k]：对每个 expert e，顺序记录路由到 e 的 token ID
    4. 构建 expert_token_offsets[E+1]：前缀和，记录每个 expert 的 token 起止位置
    5. 构建 token_expert_indices[L×k]：按 token ID 排列的 expert ID
    6. 构建 token_index_map[L×k]：每个 token 在 expert_token_indices 中的位置
    7. for each expert e_i:
         token_ids = expert_token_indices[offsets[i]:offsets[i+1]]
         x_ei = x[token_ids]  // on-the-fly gather，不 materialize
         h_ei = σ(W1_i · x_ei)  // 仅保存中间结果用于 backward
         y_ei = W2_i · h_ei
    8. for each token j:
         for each routed expert e_k:
             y_j += g_{j,k} · y_ei[token_index_map[j][k]]  // on-the-fly reduction
    9. 输出：y ∈ R^{L×d}
    关键记忆节省：不分配 L×K×d 的 routed token buffer，仅分配 4×L×K 的 int32 索引（vs bf16 激活的数百 GB）。
