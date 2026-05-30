## Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包含两部分：(1) JRT-Prompt——极简的prompting策略，将context在prompt中重复多次，使模型看到所有数据顺序，从而绕过causal模型对输入顺序的依赖。例如Î = A(C, Q, C, Q)后再生成答案；(2) JRT-RNN——受Prefix-LM启发的encoder-decoder循环架构，使用非因果的Prefix Linear Attention (PLA)处理prompt前缀（encoder区域），再因果解码输出。JRT-RNN使用两套独立的key/value投影(k_e,v_e用于encoder, k_d,v_d用于decoder)，结合NTP+MLM联合训练目标。基于Based架构（混合gated convolution + sliding window attention + linear attention），将linear attention层替换为JRT-RNN的PLA层。
  实验比较：(a) JRT-Prompt: 16个off-the-shelf循环LM（Based 1.3B, Mamba 130M/370M/1.4B/2.8B, Mamba-2 130M/370M/1.3B/2.7B, GLA 1.3B/2.7B）在6个recall-intensive ICL任务（FDA, SWDE, NQ, SQUAD, TriviaQA, Drop）上对比default vs JRT-Prompt zero-shot prompting，也对比Transformer++；(b) JRT-RNN: 360M/30B和1.3B/10B/50B参数训练，对比Based、Mamba、Transformer++在同token量下的ICL质量和SuperGLUE通用语言理解；(c) 合成SD任务: causal vs non-causal Based变体，不同state size下数据顺序敏感性验证；(d) Pile perplexity slicing (AR vs Other slices)；(e) 推理吞吐量prefill benchmark对比FlashAttention-2和多种线性注意力实现。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100-80GB GPU集群。JRT-RNN训练基于FlashAttention代码库（https://github.com/Dao-AILab/flash-attention/tree/main），使用NVidia A100-80GB。推理吞吐量benchmark：单张NVIDIA H100 GPU，prefill latency测量sequence length 2048-32768和batch size 2-64，取20次迭代平均。合成任务Based模型：论文未具体说明GPU型号（推测商用NVIDIA GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) JRT-Prompt评测：Based 1.3B (10B/50B Pile tokens)、Mamba 130M/370M/1.4B/2.8B (300B Pile tokens)、Mamba-2 130M/370M/1.3B/2.7B (300B Pile tokens)、GLA 1.3B/2.7B (100B SlimPajama tokens)、Transformer++ 1.3B (10B/50B Pile tokens)；(2) JRT-RNN训练：360M参数 (30B tokens) 和1.3B参数 (10B/50B tokens)，encoder区域长度M=1024，总序列长度N=2048。Based架构混合了gated convolution (kernel size=3)、sliding window attention (window=128)、linear attention (Taylor feature map, feature dim=16, 2nd order approximation)。JRT-RNN的PLA层feature map使用2阶Taylor近似: φ(q)^Tφ(k) = 1 + q^T k + (q^T k)²/2；(3) 合成SD任务：Based架构4层，交替gated convolution和linear attention，无位置编码。
  数据集：Pile（预训练，GPT2BPETokenizer tokenize，所有模型相同数据顺序）。Benchmark：(a) Recall-intensive ICL: FDA (FDA报告信息抽取，1102 examples/avg 1999.9 tokens)、SWDE (HTML网页信息抽取，1111/1036.1)、SQUADv2 (文档QA，2984/151.9)、Natural Questions (3157/8857.7)、TriviaQA (1698/310.1)、Drop (2084/236.6)，所有用cloze completion格式（Llama-3-70B改写）；(b) SuperGLUE: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC；(c) Pile test set perplexity slicing；(d) 合成Set Disjointness任务（|V|=2048）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/HazyResearch/prefix-linear-attention，模型权重在HuggingFace发布。JRT-Prompt评测使用LM-Eval Harness，基于HuggingFace上的开源模型权重。JRT-RNN训练代码集成在Based实现中（https://github.com/HazyResearch/based）。

  **JRT-RNN Prefix Linear Attention (PLA) 算法pipeline**:

  ```
  Input: u ∈ R^{N×d} (输入序列, N=2048, M=1024为encoder区域)
         特征图 φ: R^d → R^{d̃} (Taylor 2nd-order: φ(q)^T φ(k) = 1 + q^T k + (q^T k)²/2)

  // Encoder投影 (前M个token)
  k_e = φ(W_{ke} · u_{1:M})  // encoder key, ∈ R^{M×d̃}
  v_e = W_{ve} · u_{1:M}     // encoder value, ∈ R^{M×d}

  // Decoder投影 (全部N个token, 后N-M个causal)
  q_d = φ(W_{qd} · u)       // decoder query, ∈ R^{N×d̃}
  k_d = φ(W_{kd} · u)       // decoder key, ∈ R^{N×d̃}
  v_d = W_{vd} · u          // decoder value, ∈ R^{N×d}

  // Prefix Linear Attention (Eq.3)
  For i = 1 to N:
    num_i = q_d[i] · (Σ_{j=1}^{i} k_d[j]^T v_d[j] + Σ_{j=1}^{M} k_e[j]^T v_e[j])
    den_i = q_d[i] · (Σ_{j=1}^{i} k_d[j] + Σ_{j=1}^{M} k_e[j])
    y_i = num_i / den_i

  // Recurrent view (推理):
  // Prefill: 并行计算encoder KV-state和decoder初始state
  s_M = Σ_{j=1}^{M} (k_e[j]^T v_e[j] + k_d[j]^T v_d[j])  // KV-state ∈ R^{d×d̃}
  z_M = Σ_{j=1}^{M} (k_e[j] + k_d[j])                    // K-state ∈ R^{d̃}

  // Decoding (i > M, 每token O(1)):
  s_i = s_{i-1} + k_d[i]^T v_d[i]
  z_i = z_{i-1} + k_d[i]
  y_i = (q_d[i] · s_i) / (q_d[i] · z_i)
  ```

  **训练目标 (Eq.5)**:
  ```
  L = (w1 * L_NTP + w2 * L_MLM) / (w1 + w2)
  // L_NTP: 标准next token prediction loss, 仅计算causal区域 {u_M..u_N}
  // L_MLM: masked language modeling loss, 随机mask encoder区域比例P的token
  // 推理时不使用[MASK] token
  ```

  **JRT-Prompt策略**:
  ```
  // 标准ICL: Ŷ = A(C, Q)
  // JRT-Prompt: Ŷ = A(C, Q, C, Q)
  // 第二轮时模型以完整context view决定存储什么信息
  ```

  JRT-RNN与标准decoder-only循环LM的关键差异：
  - 使用两套独立KV投影（encoder和decoder分开），而非Prefix-LM的单套共享投影
  - 训练时encoder区域额外计算MLM loss（比例P mask），Prefix-LM仅计算NTP loss
  - Encoder区域使用非因果sum（非causal cumsum），让encoder区域token互相可见
  - prefill阶段的recurrent state初始化包含encoder的非因果KV-state (Eq.4)
  - 解码步骤与标准causal linear attention完全相同 (Eq.2)，无额外开销

  JRT-Prompt效率分析：虽将context长度翻倍(2N vs N)，但使用sub-quadratic循环架构仍渐进优于Transformer的O(N²)。N=32768, batch=16, H100上Based+JRT-Prompt提供11.9×于FlashAttention-2的prefill吞吐量。
