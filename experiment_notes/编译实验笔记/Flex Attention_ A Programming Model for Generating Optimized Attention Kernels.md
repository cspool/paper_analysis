## Flex Attention: A Programming Model for Generating Optimized Attention Kernels

- 属于编译框架的实现是什么？实验比较什么？
  FlexAttention 是一个编译器驱动的编程模型，允许用户在几行 PyTorch 代码中定义注意力变体（通过 score_mod 和 mask_mod callable），由编译器自动生成优化的 Triton attention kernel。核心编译框架实现包括：(i) 统一抽象——将注意力变体分为 score modification（score_mod）和 attention mask（mask_mod）两种模式，score_mod 基于位置修改 score 标量（如 Alibi bias），mask_mod 指定哪些 score 设为 -inf（如 causal mask）；(ii) 模板化 lowering（Template-based Lowering）——利用 TorchDynamo 捕获 score_mod/mask_mod 的计算图，TorchInductor 将子图翻译为 Triton 代码，动态注入到手写的 forward/backward/decoding 三个 attention kernel 模板中；(iii) 逻辑融合（Logical Fusion）——通过 and_mask 和 or_mask 支持 mask 设计的组合，解决注意力变体的组合爆炸问题；(iv) 自动 backward pass 生成——通过 torch.autograd 构建 forward 和 backward 计算图。

  实验比较：(1) 7 种注意力变体（noop, causal, alibi_bias, sliding_win, prefix_lm, soft_cap, document_mask）的 kernel 性能 vs FAv2、FAv3、SDPA（cuDNN, mem_efficient）、FAKV；(2) 端到端训练性能——torchtune fine-tuning LLaMa3-8B on Alpaca vs SDPA；(3) 端到端推理性能——gpt-fast 上 LLaMa3.1-8B 和 70B vs SDPA；(4) Paged Attention 场景 overhead vs FlashAttn-v2。结果：training forward 1.00×-1.22× vs FAv2 causal, backward 0.86×-1.05×；对 FAv2 不支持的变体 5.49×-8.00× vs SDPA；推理 0.93×-1.45× vs FAKV；端到端训练 2.4×, 推理 2.04× speedup；Paged Attention 额外 overhead <1%。

- 硬件平台是什么，配置是什么。
  Nvidia H100 GPU（功率限制 650W，内存带宽限制 2.4TB/s），Nvidia A100 GPU（功率限制 330W），Nvidia A6000 GPU。KV size 固定 256 MiB，head dimension 64，数据类型 bfloat16。

- 开源编译框架是什么。修改了什么。
  基于 PyTorch 核心编译栈（TorchDynamo + TorchInductor + torch.compile）和 Triton（https://github.com/triton-lang/triton）构建。FlexAttention 已合并入 PyTorch 主仓库（pytorch/pytorch），在 PyTorch 2.5+ 中可用。示例和工具在 attention-gym（https://github.com/pytorch-labs/attention-gym）中提供。
  
  修改/新增内容：
  - 前端 API：`torch.nn.attention.flex_attention.flex_attention()` 和 `create_block_mask()`，接受 score_mod 和 mask_mod callable
  - 手写 Triton attention kernel 模板（forward/backward/decoding），模板内预留 score_mod 和 mask_mod 代码注入点
  - TorchDynamo 捕获 + TorchInductor lowering pipeline：将用户定义的 score_mod/mask_mod PyTorch 函数翻译为 Triton 代码块，动态注入模板
  - BlockMask 数据结构与 create_block_mask 工具（通过 torch.vmap 自动生成 block-level sparsity）
  - 逻辑融合支持（and_mask/or_mask）

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  FlexAttention 已集成至 PyTorch 2.5+ 核心（https://github.com/pytorch/pytorch），attention-gym（https://github.com/pytorch-labs/attention-gym）提供示例和工具。

  编译框架完整流程（以 causal mask + sliding window attention 为例）：

  1. **用户输入**：用户用 PyTorch 定义 mask_mod 和 score_mod 函数：
  ```python
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  
  def causal_mask(b, h, q_idx, kv_idx):
      return q_idx >= kv_idx
  
  def sliding_window(b, h, q_idx, kv_idx):
      return q_idx - kv_idx <= 1024
  
  # 逻辑融合
  from torch.nn.attention.flex_attention import and_masks
  combined_mask = and_masks(causal_mask, sliding_window)
  
  # 生成 BlockMask
  block_mask = create_block_mask(combined_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192)
  
  # 调用 FlexAttention
  output = flex_attention(query, key, value, block_mask=block_mask)
  ```

  2. **TorchDynamo 图捕获**：torch.compile 的 TorchDynamo 拦截 flex_attention 调用，捕获 score_mod 和 mask_mod 的 PyTorch 计算图（通常非常轻量，主要是element-wise的位置比较和算术操作）。

  3. **TorchInductor lowering**：TorchInductor 将捕获的 score_mod/mask_mod 子图翻译为 Triton 原语操作（Triton code blocks）。同时编译 forward 和 backward 两个方向的子图。

  4. **模板注入与 kernel 生成**：Triton 代码块被动态注入到 3 个手写的 Triton attention kernel 模板中：
     - **Forward 模板**：包含 online softmax + tiled QK^T GEMM + tiled PV GEMM 的标准 fused attention 优化，score_mod 代码块注入到 QK^T GEMM 之后、softmax 之前对每个 score tile 执行，mask_mod 代码块注入到部分 block 的 element-wise masking 位置。
     - **Backward 模板**：通过 torch.autograd 生成 score_mod 和 mask_mod 的反向计算图，注入到 backward attention kernel 模板。
     - **Decoding 模板**：针对推理场景（q_len=1）的专用模板。

  5. **BlockMask sparsity 优化**：create_block_mask 在编译时利用 torch.vmap 对 mask_mod 进行向量化评估，将 score 矩阵按 block（默认 block_size=128）分割，生成两个张量：
     - `kv_num_blocks` [B, H, Num_Row]：每行非零 block 数
     - `kv_indices` [B, H, Num_Row, Num_Col]：非零 block 的列索引
     进一步区分 Full Blocks（全部 score 可见，跳过 mask_mod）和 Partial Blocks（部分 masked，需运行时逐元素执行 mask_mod），获得约 15% 额外性能提升。

  6. **运行时执行**：生成的 Triton kernel 在 GPU 上执行。SM 沿 Q_LEN 维度切分 tile 并行处理，每个 SM 沿 KV_LEN 维度迭代。通过 kv_indices 实现间接内存访问，跳过完全 masked 的 block。数据预取 pipeline：当前 score block 计算时，下一 KV tile 从 HBM 预取到 SRAM。

  7. **输出**：标准的 attention output tensor，与 PyTorch SDPA API 兼容。
