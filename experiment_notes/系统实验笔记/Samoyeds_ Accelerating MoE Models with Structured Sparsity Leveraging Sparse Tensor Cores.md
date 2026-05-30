## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在SOTA serving框架（vLLM 0.4.0.post1 + Transformers v4.40.0）中集成Samoyeds sparse-sparse kernel，替代MoE层的标准GEMM执行流程。核心修改：(1) 消除input permutation开销——标准MoE流程需将tokens permute到各expert的tensor（产生额外内存分配和数据搬运），Samoyeds通过SEL选择数组在kernel内部直接索引有效tokens，跳过显式permutation；(2) 消除output un-permutation开销——expert输出从register直接写入压缩layout，避免先写global memory再读回的roundtrip；(3) operator fusion——将activation function与前驱operator融合，将weighted accumulation（scalar广播+点乘）与矩阵乘法融合，减少kernel launch和中间结果materialize；(4) 压缩output layout——MoE中间结果（expert输出）在accumulation前是row-wise稀疏的（稀疏比=expert数量），Samoyeds仅输出非零行，避免传输零值。batch size支持能力显著提升（平均4.41×）。
  - 实验比较：(a) MoE层级别：对比Transformers（v4.40.0）、MegaBlocks、vLLM-DS（含fused MoE kernel，2024年3月合并版本，~2.8× speedup over non-fusion），评估两类MoE（带/不带shared experts）的speedup；(b) 端到端模型级别：6种MoE模型（Qwen2-MoE/DeepSeek-MoE/MiniCPM-MoE/OpenMoE-34B/Mixtral-8×7B/Mixtral-8×22B）的latency speedup（seq_len=4096，batch=1或16）；(c) 不同batch size下吞吐量对比；(d) 最大batch size支持对比。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA GeForce RTX 4070 Super
  - CPU：Intel i7-12700，16G×2 DDR5，Ubuntu 22.04LTS
  - 软件栈：CUDA 12.1, cuSPARSELt 0.4.0, PyTorch 2.1.0, Transformers v4.40.0, vLLM 0.4.0.post1
  - 功耗控制：所有实验禁用CPU frequency scaling

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace Transformers v4.40.0 + vLLM 0.4.0.post1（含fused MoE kernel PR #2453）
  - 修改内容：将MoE decoder layer中的expert计算（gate_proj, up_proj, down_proj三个线性层）替换为Samoyeds sparse-sparse kernel调用。具体地：(a) 在vLLM的fused MoE kernel基础上，替换内部GEMM为Samoyeds kernel（处理双端稀疏）；(b) 消除expert输入/输出的permute/un-permute操作（原本由vLLM fused kernel管理）；(c) 集成weighted accumulation fusion和activation fusion。这些修改通过pybind11暴露的Python模块与框架对接。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/guqiqi/Samoyeds.git，Docker镜像可直接运行end-to-end实验
  - 框架输入→硬件执行全过程（单个decoder layer推理）：
    ```
    输入：prompt tokens [batch_size × seq_len]，路由权重
    
    1. Attention层（FlashAttention2）：
       输入tokens → Q/K/V projection → FlashAttention → output
       
    2. MoE层（Samoyeds优化路径）：
       a) Router: tokens → routing scores → top-k expert assignment
          - Qwen2-MoE/DeepSeek-MoE: 含shared experts（所有token都经过）
          - Mixtral-8×7B/22B: 仅routed experts
       b) Expert计算（Samoyeds kernel）：
          对每个expert E_i:
            - SEL[i] = {j | token[j] routed to E_i}  # 隐式permutation（无内存拷贝）
            - 加载编码权重: data[i], indices[i], metadata[i]（已在offline阶段压缩为Samoyeds格式）
            - gate_proj: C_gate = Samoyeds_spmm(W_gate_encoded, input[SEL])
            - up_proj: C_up = Samoyeds_spmm(W_up_encoded, input[SEL])
            - Activation fusion: C_act = SiLU(C_gate) * C_up  # fused in-kernel，无中间materialize
            - down_proj: C_out = Samoyeds_spmm(W_down_encoded, C_act)
            - 输出以压缩layout写入GMEM（仅非零行）
       c) Weighted accumulation（fused）:
          对每个token t:
            output[t] = Σ_i router_score[t][i] * expert_output[i][t]  # fused in-kernel
        
       Pipeline视图：
       GMEM → [cp.async] → SMEM (A_tile, B_tile) → [ldmatrix] → Register → [mma.sp] → SpTC计算 → Register C → [shuffle/stationary] → [store] → GMEM (压缩output)
    ```
  - 关键性能增益：消除input permutation overhead（大expert数量的模型收益更大，如Qwen2-MoE 60 experts和DeepSeek-MoE 64 experts），消除output zero-value传输（高sparsity模型加速达2.66×），fused activation+accumulation减少kernel launch和内存roundtrip。
