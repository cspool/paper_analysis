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
