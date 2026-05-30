## xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Fused Generation Kernels for mLSTM Cell。在自回归生成时，mLSTM cell 的 recurrent 公式（Eq. 1-9）包含 outer-product、多个 dot-product 和逐点操作（pointwise operations），在标准实现中会分解为多个独立的 GPU kernel 调用。每个 kernel 调用都需要从 GPU HBM 加载输入并存储输出，增加了慢速内存操作的占比。论文开发了 fused GPU kernels（Triton 编写），将这些中间结果保持在 GPU 的 compute chip（SRAM/register）上，避免不必要地传输到 GPU 内存（HBM）。此外，论文的 chunkwise-parallel 训练 kernel 基于 FlashLinearAttention 技术（Yang et al., 2024b），在训练时对序列分块并行处理。
  - 实验比较：在 Fig. 4-7 中通过整体推理速度 benchmark 间接评估 kernel 效果。xLSTM 7B 比 Falcon-Mamba（Mamba 1）和 Codestral-Mamba（Mamba 2）快约 50% 的生成吞吐，在 prefill 长度 0 时甚至快于 Llama 系列 Transformer；在 65536 token prefill 吞吐测试中，xLSTM 7B 比 Codestral Mamba 高约 70%。论文未对单个 fused kernel 进行独立的 micro-benchmark。

- 后端平台是什么，配置是什么。
  - GPU: 单 NVIDIA H100 GPU
  - 推理精度：论文未明确说明（从 HuggingFace model card 可推断为 bfloat16）
  - 推理框架：HuggingFace transformers + torch.compile + PyTorch CUDA Graphs
  - 训练：128× NVIDIA H100 GPU，FSDP + activation checkpointing

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：使用 HuggingFace transformers 库中的各模型实现，统一用 torch.compile 和 PyTorch CUDA Graphs 优化后测量
  - Kernel 代码开源在 https://github.com/NX-AI/mlstm_kernels（Triton-based）
  - 修改内容：将 mLSTM cell 在生成模式下的多个独立 GPU kernel（outer-product, dot-product, pointwise ops）融合为单个 fused kernel，减少 HBM 读写次数；训练时使用 chunkwise-parallel kernel 替代 naive 实现
  - 对比对象：Llama-2-7B、Llama-3.1-8B（attention-based）、Falcon-Mamba-7B（Mamba 1 architecture）、Codestral-Mamba-7B（Mamba 2 architecture）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - Triton fused kernel 开源：https://github.com/NX-AI/mlstm_kernels
  - 模型实现开源：https://github.com/NX-AI/xlstm（PyTorch）和 https://github.com/NX-AI/xlstm-jax（JAX）
  - **Fused Generation Kernel 原理**：
    1. **输入**：当前 token 的 input vector x_t ∈ R^d，上一时间步的 recurrent state (C_{t-1}, n_{t-1}, m_{t-1}, h_{t-1})
    2. **Kernel 执行流程（单次 fused kernel 调用）**：
       - Gate computation: 计算 i_tilde, f_tilde（scalar soft-capped pre-activations）和 o_tilde（vector output gate pre-activation）
       - Max state update: m_t = max(log(σ(f_tilde)) + m_{t-1}, i_tilde)
       - Gate activation: f_t = exp(log(σ(f_tilde)) + m_{t-1} - m_t), i_t = exp(i_tilde - m_t)
       - Memory Update (in SRAM): C_t = f_t * C_{t-1} + i_t * (k_t^T v_t)（outer product 在片上计算）
       - Normalizer Update: n_t = f_t * n_{t-1} + i_t * k_t
       - Hidden State Retrieval: h_tilde = C_t^T @ q_norm / max(|n_t^T @ q_norm|, exp(-m_t))
       - Output: h_t = o_t ⊙ Norm(h_tilde)
    3. **输出**：当前时间步的 hidden state h_t 和更新后的 recurrent state (C_t, n_t, m_t)
    4. **关键优化**：所有中间结果（k_t, v_t, q_t, gate values, C_t 更新中的 outer product 结果）都在 GPU SM 上的 SRAM/Register file 中保持，不写回 HBM。只有最终的 h_t 和 state 写回。由于 mLSTM 不使用 softmax attention，没有 QK^T 矩阵的全序列计算，每次 recurrent step 的 FLOPs 恒定。
  - **评估原理**：
    1. 使用 HuggingFace transformers 加载各模型
    2. 用 torch.compile 对模型计算图进行 JIT 编译优化
    3. 用 PyTorch CUDA Graphs 捕获重复的推理步骤，消除 kernel launch overhead
    4. 在单 H100 GPU 上，batch size 1，测量：(a) 在不同 prefill 长度（0 到 128K tokens）下生成 100 token 的吞吐（tokens/sec）；(b) 在不同生成长度下的生成时间和 GPU 内存占用；(c) Time To First Token 延迟；(d) 在 65536 token 下不同 batch size 和 context length 的 prefill 吞吐

## RWKV__Reinventing_RNNs_for_the_Transformer_Era

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Custom CUDA kernel 用于 WKV 计算。由于 WKV 的逐元素时间依赖计算（`wkv_t = Σe^{-(t-1-i)w+k_i}⊙v_i / Σe^{-(t-1-i)w+k_i}`）在标准深度学习框架中是串行的，论文开发了自定义CUDA kernel来在GPU上高效执行这一计算。其余部分（矩阵乘法、逐点运算）本身已可并行化。
  - 实验比较：论文未对kernel本身进行独立性能benchmark。整体推理性能在Section 6和Appendix K中通过比较RWKV与Transformer家族（BLOOM、OPT、GPT-Neo、Pythia）在文本生成中的时延（s）和内存（CPU RAM、GPU VRAM）来间接评估。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A100 80 GB
  - CPU: x86
  - 推理精度: float32（使用HuggingFace Transformers）

- 评估性能的软件/脚本是什么。修改了什么。
  - 推理框架: HuggingFace Transformers (Wolf et al., 2020)
  - 论文对比了RWKV各规模模型（169M-14B）与BLOOM (560M-3B)、OPT (125M-13B)、GPT-Neo (125M-2.7B)、Pythia (160M-12B)在相同硬件上的文本生成推理时延和内存使用。
  - 论文未说明完全使用CUDA kernel进行推理时的benchmark；HuggingFace Transformers路径可能使用PyTorch原生实现而非自定义CUDA kernel。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/BlinkDL/RWKV-LM（包含CUDA kernel的C++/CUDA实现，位于仓库的cuda/目录）
  - 论文说明：自定义CUDA kernel将WKV的串行扫描在GPU上并行化。kernel的工作原理：WKV计算公式涉及指数加权移动平均（指数衰减权重e^{-w}乘以历史K,V值），这本质上是一个前缀和扫描操作。标准PyTorch实现因每次计算wkv_t需要遍历1..t-1而低效。CUDA kernel通过work-efficient parallel scan算法（如Blelloch scan），将串行O(T)的前缀和分解为O(log T)并行步骤，在GPU上并发处理batch和channel维度。
  - Kernel输入到输出过程：
    1. 输入：k_tensor [B, T, d], v_tensor [B, T, d], w [d]（通道级时间衰减）, u [d]（当前token bonus）
    2. CUDA kernel在GPU上为每个(batch, channel)对执行并行前缀扫描
    3. 输出：wkv_tensor [B, T, d]——每个位置t的加权平均V值
    4. 后续output gating: o_t = W_o @ (σ(r_t) ⊙ wkv_t)，通过标准cuBLAS矩阵乘法完成
  - 论文还指出可以使用更高级的parallel scan方法（Martin and Cundy, 2017）对极长序列进一步并行化。

## Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Hardware-Aware Selective Scan——用于高效计算选择性SSM（S6）在GPU上的custom CUDA kernel。核心设计基于三种经典技术：kernel fusion、parallel scan和recomputation。由于selective SSM失去了与convolution的等价性（无法使用FFT），必须用recurrent模式计算，但直接materialize expanded state h∈R^{B×L×D×N}会导致IO瓶颈。硬件感知算法方案：(1) 不将scan输入(A_bar, B_bar)∈R^{B×L×D×N}写入HBM，而是直接从慢HBM加载(Δ, A, B, C)到快SRAM；(2) 在SRAM中完成discretization和recurrence/scan；(3) 仅将最终输出y∈R^{B×L×D}写回HBM；(4) 使用work-efficient parallel scan(Blelloch 1990)并行化recurrence；(5) backward pass不存储中间states，通过recomputation从HBM重新加载输入并重算。当seqlen超过SRAM容量时分chunk处理，保留intermediate scan states跨chunk延续。
  实验比较：(a) Scan Operation Speed(Figure 8 Left)：Mamba efficient scan vs standard PyTorch scan、FlashAttention-2(attention)、PyTorch convolution(FFT-based)，变seqlen 512-512K，D=1024, N=16, BF16, A100 80GB PCIe, batch=1——efficient scan 40× faster than standard scan, faster than FlashAttention-2 beyond seqlen 2K；(b) End-to-End Inference Throughput(Figure 8 Right)：Mamba 1.4B/6.9B vs Transformer 1.3B/6.7B，变batch size 1-128, prompt=2048, gen=128——Mamba 4-5× higher throughput(no KV cache, higher batch sizes)；(c) Memory Benchmark(Table 15)：125M model training memory, Mamba vs Transformer(w/ FlashAttention-2), seqlen=2048, batch=1-32——Mamba memory comparable to optimized Transformer(4.8GB vs 4.6GB at batch=1)。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU。Benchmark配置：batch size=1 for scan speed, 模型维度D=1024, state dimension N=16, BF16精度。Inference throughput: prompt length=2048, generation length=128, batch size 1-128, 测量3次取平均。Memory benchmark: 125M model, seqlen=2048, batch size 1-32 on 1 A100 80GB。

- 评估性能的软件/脚本是什么。修改了什么。
  自研fused selective scan CUDA kernel。Baseline对比：standard PyTorch parallel scan(无kernel fusion, 需materialize A_bar/B_bar/C in HBM)、PyTorch convolution(FFT-based, O(L log L))、FlashAttention-2(causal mask, Dao 2024)。修改内容：用fused kernel替代标准实现——将discretization、parallel scan和C乘法合并为一个CUDA kernel，在SRAM中完成全部计算，避免反复HBM↔SRAM的O(BLDN) IO传输。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  完全开源：https://github.com/state-spaces/mamba (Apache 2.0)，包含selective scan CUDA kernel实现。

  **Fused Selective Scan Kernel执行原理** (Section 3.3.2 + Appendix D):

  ```
  # 标准实现 (inefficient):
  # Step 1: 将Δ, A, B, C从HBM读到SRAM → discretize → A_bar, B_bar of size (B,L,D,N)
  # Step 2: 将A_bar, B_bar写回HBM
  # Step 3: 从HBM读A_bar, B_bar, C → parallel scan → 写结果回HBM
  # Step 4: 从HBM读scan结果 → 乘C → 写最终输出y回HBM
  # 总IO: O(BLDN) — 受state dim N (≈16) 放大

  # Fused kernel (efficient):
  # 1. 从慢HBM加载原始参数 (Δ, A, B, C) 到快SRAM
  #    — O(BLD + D*N) bytes, 不含N因子!
  # 2. 在SRAM中执行discretization:
  #    A_bar = exp(Δ ⊙ A)  → (B,L,D,N)
  #    B_bar = Δ ⊗ B       → (B,L,D,N)
  # 3. 在SRAM中执行parallel associative scan
  #    — 使用work-efficient Blelloch scan并行化
  #    — 中间states h ∈ R^{B×L×D×N} 仅存在于SRAM
  # 4. 在SRAM中乘C并累加: y_t = Σ C_t * h_t
  #    — 最终输出 y ∈ R^{B×L×D}
  # 5. 将最终输出y写回HBM
  #    — O(BLD) bytes, 无N因子
  # IO减少因子: O(N), N=16 → 16× less HBM traffic
  ```

  **Chunked scan处理长序列** (当seqlen超出SRAM容量):
  ```
  # 将序列分为chunks:
  For each chunk:
    在SRAM中fused scan该chunk
    保存最后一个time step的intermediate scan state
  # 将intermediate state传给下一chunk作为初始state
  # 保证全局scan的正确性
  ```

  **Recomputation策略** (backward pass):
  ```
  Forward: 不保存中间states h ∈ R^{B×L×D×N} (节省IO但需要backward重算)
  Backward: 从HBM重新加载输入(Δ, A, B, C)和output gradient(∈R^{B×L×D})
            → 在SRAM中重算intermediate states
            → 计算输入梯度(Δ_grad, A_grad, B_grad, C_grad)
            → 写回HBM
  # 总IO: O(BLD + D*N) input + O(BLD + D*N) gradient = 2(BLD + D*N)
  # vs 存储intermediates: O(BLDN) — recomputation wins when N > 2
  ```

  **Memory efficiency vs FlashAttention:**
  ```
  # Per-layer activation memory (BF16 training):
  # - FlashAttention: ~12 bytes/token
  # - MLP (Transformer): ~20 bytes/token
  # - Selective SSM (Mamba): ~16 bytes/token
  # 2×Mamba blocks ≈ 1×Attention + 1×MLP (both ~32 bytes/token)
  ```

  实测benchmark(Figure 8, A100 80GB):
  ```
  Scan Speed (seqlen=32K):
    Mamba efficient scan: ~0.15ms
    FlashAttention-2 (causal): ~1.0ms (6.7× slower)
    Standard PyTorch scan: ~6.0ms (40× slower)
    PyTorch convolution: ~0.3ms

  Inference Throughput (Mamba-1.4B vs Transformer-1.3B):
    Batch=64: Mamba ~3000 tokens/s, Transformer ~700 tokens/s (4.3×)
    Mamba-6.9B > Transformer-1.3B throughput despite 5× larger!
  ```

## Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是基于Based CUDA kernel（来自ThunderKittens, https://github.com/HazyResearch/ThunderKittens）的扩展，支持JRT-RNN的Prefix Linear Attention (PLA) 的IO-aware prefill计算。原始Based kernel在prefill阶段通过在warp-register间分区存储大型矩阵值recurrent state（KV-state ∈ R^{d×d̃}），实现高效的线性注意力计算。JRT-RNN扩展（Algorithm 2）：(1) 第一次调用fnbased(k_e, v_e)计算encoder的KV-state（使用非因果sum而非causal cumsum），结果存储在寄存器A0, A1, A2中（对应Based Taylor近似的0阶、1阶、2阶项）；(2) 第二次调用fnbased(q_d, k_d, v_d)从encoder初始化的register state开始计算decoder输出y，并写入SRAM；(3) 最终y从SRAM写回HBM。相比JRT-Prompt需要2× prefill时间（重复context），JRT-RNN仅需1.24× Based prefill时间。
  实验比较JRT-RNN Custom CUDA、JRT-Prompt Custom CUDA vs FlashAttention-2、Based Custom CUDA、Based Triton (FLA)、Based PyTorch、Fast Transformer CUDA、JRT-RNN PyTorch，在H100上测量prefill latency (ms)，变sequence length (2048-32768) 和batch size (2-64)。JRT-RNN CUDA达19.2×于FA2和22.0×于FLA Triton的吞吐量（N=32768, B=16）；batch size=64时9.7×于FA2和11.5×于FLA。

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU。Benchmark配置：sequence length 2048/4096/8192/16384/32768（固定batch=16），batch size 2/4/8/16/32/64（固定seqlen=16384）。每个测点20次迭代平均。模型基于1.3B参数Based架构。所有比较在同一H100上进行。

- 评估性能的软件/脚本是什么。修改了什么。
  Custom CUDA kernel（从ThunderKittens Based kernel修改而来）。Baseline对比：(a) FlashAttention-2 (softmax attention CUDA fusion kernel)；(b) Fast Transformer CUDA kernel（线性注意力开源CUDA实现）；(c) FLA Triton kernel（Flash Linear Attention库中Based的Triton并行实现）；(d) PyTorch实现。
  修改内容：在Based kernel基础上，(1) 避免第一次调用时与queries相乘，仅计算KV-state；(2) 使用encoder序列的最终行（row M）作为KV-state的初始值传递给decoder；(3) encoder部分使用非因果sum替代causal cumsum；(4) 增加encoder的FLOPS: BMHD（feature map计算）+ 3BMHdD（k_e·v_e点积+D维度求和+与decoder state相加）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  Custom CUDA kernel开源：https://github.com/HazyResearch/ThunderKittens（Based kernel）和 https://github.com/HazyResearch/prefix-linear-attention（JRT-RNN扩展）。评估代码在prefix-linear-attention仓库。

  **JRT-RNN CUDA Kernel执行原理** (Algorithm 2):

  ```
  Input: q_d, k_d, v_d ∈ R^{N×d} (decoder), k_e, v_e ∈ R^{M×d} (encoder)
  Output: y ∈ R^{N×d}

  Step 1: 初始化SRAM buffer和register file fragments
    - 寄存器A0, A1, A2: 存储KV-state (对应Taylor展开0/1/2阶项)
    - SRAM buffer y: 存储最终输出

  Step 2: 运行 fnbased(k_e, v_e) → encoder KV-state
    - 使用非因果sum (而非causal cumsum)
    - 不乘以queries (与原Based算法不同)
    - 结果保留在寄存器A0, A1, A2中
    - KV-state = Σ_{j=1}^{M} (k_e[j]^T · v_e[j])

  Step 3: 运行 fnbased(q_d, k_d, v_d) → decoder输出
    - 从Step 2的register state初始化
    - 对decoder区域执行causal cumsum
    - KV-state_dec = encoder_state + Σ_{j=1}^{i} (k_d[j]^T · v_d[j])
    - K-state_dec = encoder_k_sum + Σ_{j=1}^{i} k_d[j]
    - y_i = (q_d[i] · KV-state_dec) / (q_d[i] · K-state_dec)
    - y写入SRAM

  Step 4: Store y from SRAM → HBM
  ```

  **FLOPS分析** (per layer):
  ```
  Baseline causal LA: 2BNHD (qd/kd feature map) + 4BNHdD (kd·vd dot + cumsum + qd dot + D sum)
  PLA增加: BMHD (ke feature map) + 3BMHdD (ke·ve dot + D sum + encoder/decoder state merge)
  PLA memory (decode): 与causal LA相同 (recurrent state size不变)
  ```

  **Prefill latency测量 (Table 5, H100)**:
  ```
  N=32768, B=16:
    FA2: 107.8ms → JRT-RNN CUDA: 5.6ms (19.2× faster)
    FLA Triton: 123.7ms → JRT-RNN CUDA: 5.6ms (22.0× faster)
    JRT-Prompt CUDA (2× N): 9.0ms → 11.9× > FA2, 13.7× > FLA

  B=64, N=16384:
    FA2: 108.2ms → JRT-RNN CUDA: 11.1ms (9.7× faster)
    FLA Triton: 127.8ms → JRT-RNN CUDA: 11.1ms (11.5× faster)
  ```

  Based kernel的核心优化是避免HBM↔SRAM反复传输，将矩阵值KV-state (∈ R^{d×d̃}，对于d=2048, d̃=273可达数MB) 分片存储在warp registers中，实现SRAM-resident state管理。PLA扩展通过复用已有register管理逻辑——先用encoder输入填充register state，再切换到decoder计算——几乎零overhead地支持了encoder-decoder模式。

## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Finch（RWKV-6）WKV计算的custom CUDA kernel，用于训练时加速。核心设计：不使用time-parallel的associative scan方法（虽高度并行但涉及反复HBM↔SRAM transfer），而是沿非时间维度并行，将state操作保持在fast SRAM中以减少memory transfer开销。具体而言，Finch的WKV计算可通过式(19)沿序列做并行prefix-sum，也可按式(21)-(22)以recurrent方式计算——论文选择后者，搭配SRAM-resident state management的CUDA kernel。
  实验对比Finch kernel vs Mamba kernel (2× pass, 模拟同层数) vs Flash Attention v2 (PyTorch实现)：(a) Memory Usage vs Sequence Length (A100 80GB, batch=8, D=4096, head=64, Mamba D=8192/state=16) — 图6；(b) Time vs Sequence Length — 图7。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB GPU。Benchmark配置：batch size=8, model dimension=4096, head size=64 (Finch/Flash Attention), state dimension=16/model dim=8192 (Mamba, expansion factor=2)。Finch kernel不做time维度并行，选择沿非时间维度并行+SRAM state residency。

- 评估性能的软件/脚本是什么。修改了什么。
  自研Finch CUDA kernel。对比baseline：Mamba kernel（2× pass模拟与Transformer同层数），Flash Attention v2 (PyTorch实现)。修改：为Finch的新WKV计算（data-dependent time-varying decay w_t, matrix-valued state）编写custom CUDA kernel，核心优化是避免HBM↔SRAM反复传输，state操作保持在SRAM。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  训练代码开源：https://github.com/RWKV/RWKV-LM。

  评估原理：对预训练forward pass的实际wall-clock time和peak GPU memory进行测量：
  ```
  配置: batch_size=8, model_dim=4096, head_size=64
        序列长度从 256 扫至 16384

  Finch kernel:
    输入: x_t ∈ R^{B×H×D/H}, s_{t-1} ∈ R^{B×H×D/H×D/H}
    计算: 
      - Token shift (ddlerp): 在SRAM中计算, A∈R^{D×32}, B∈R^{32×D}
      - WKV: k_t^T · v_t → 沿非时间维度并行, s_t = diag(w_t)·s_{t-1} + k_t^T·v_t
      - state操作保持在SRAM，不反复写入HBM
      - 输出: o_t ∈ R^{B×H×D/H}, s_t ∈ R^{B×H×D/H×D/H}

  关键结果:
    - 训练时间: Finch O(N)线性扩展（与Mamba类似），16k序列时比Flash Attention快约4.2×
    - 内存: Finch比Flash Attention省约40%，比Mamba省约17%
    - 序列长度<4k时Flash Attention更快，>4k后Finch领先（因为Flash Attention内存压力更大）
  ```
