## DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **(1) Multi-head Latent Attention (MLA)**：通过低秩键值联合压缩将 KV cache 压缩为 latent vector，同时使用解耦 RoPE 策略规避 RoPE 与低秩压缩的不兼容问题，使得 KV cache 仅为 MHA 的约 4%，但性能优于 MHA。**(2) DeepSeekMoE**：采用细粒度专家分割（fine-grained expert segmentation）和共享专家隔离（shared expert isolation），配合 Device-Limited Routing、三层辅助损失（Expert-Level / Device-Level / Communication Balance Loss）和 Token-Dropping Strategy，在保证负载均衡的同时实现经济的训练。实验比较：(a) MLA vs MHA/GQA/MQA 的 KV cache 开销和 benchmark 性能；(b) DeepSeek-V2 (236B total/21B activated) vs DeepSeek 67B (dense)、Qwen1.5 72B、LLaMA3 70B、Mixtral 8x22B 的综合 benchmark 性能；(c) 训练成本（GPU hours/trillion tokens）和推理吞吐（tokens/sec on 8xH800）。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU 集群。每个节点 8 张 H800 GPU，节点内通过 NVLink 和 NVSwitch 互联，节点间通过 InfiniBand 互联。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-V2 (60层 Transformer, hidden dim=5120, 128 heads, d_h=128, d_c=512, d_c'=1536, d_h^R=64, 2 shared + 160 routed experts, K_r=6, 236B total/21B activated)。数据集：8.1T tokens 双语（中英文约 12% 中文更多）预训练语料，BBPE tokenizer，vocab size=100K。Benchmarks：MMLU, C-Eval, CMMLU, HellaSwag, PIQA, ARC (Easy/Challenge), BBH, TriviaQA, NaturalQuestions, RACE (Middle/High), DROP, C3, CMRC, WinoGrande, CLUEWSC, Pile-test, CHID, CCPM, GSM8K, MATH, CMath, HumanEval, MBPP, CRUXEval, AGIEval。Chat 版本额外评估：MT-Bench, AlpacaEval 2.0, AlignBench, IFEval, LiveCodeBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，checkpoints 发布于 https://github.com/deepseek-ai/DeepSeek-V2。

  **MLA 算法流程（以推理阶段单 token 为例）**：
  ```
  Input: h_t ∈ R^d (第 t 个 token 的 hidden state, d=5120)
  
  // 1. Query 低秩压缩（训练时减少激活内存）
  c_t^Q = W^{DQ} @ h_t           // [d_c' × d] @ [d × 1], d_c'=1536
  q_t^C = W^{UQ} @ c_t^Q         // [d_h*n_h × d_c'] @ [d_c' × 1], d_h*n_h=128*128=16384
  q_t^R = RoPE(W^{QR} @ c_t^Q)   // [d_h^R*n_h × d_c'] @ [d_c' × 1], d_h^R*n_h=64*128=8192
  q_{t,i} = concat(q_{t,i}^C, q_{t,i}^R)  // per-head: [128+64] = [192]
  
  // 2. KV 低秩联合压缩（核心：大幅减少 KV cache）
  c_t^{KV} = W^{DKV} @ h_t       // [d_c × d] @ [d × 1], d_c=512（仅此需缓存！）
  k_t^C = W^{UK} @ c_t^{KV}      // 推理时 W^{UK} 被吸收进 W^{UQ}
  v_t^C = W^{UV} @ c_t^{KV}      // 推理时 W^{UV} 被吸收进 W^O
  k_t^R = RoPE(W^{KR} @ h_t)     // 解耦的 RoPE key，也需缓存
  k_{t,i} = concat(k_{t,i}^C, k_t^R)  // per-head: for attention
  
  // 3. Attention 计算
  o_{t,i} = sum_{j=1..t} Softmax(q_{t,i}^T @ k_{j,i} / sqrt(d_h+d_h^R)) * v_{j,i}^C
  u_t = W^O @ concat(o_{t,1}, ..., o_{t,128})
  
  // KV cache per token: (d_c + d_h^R) * l = (512 + 64) * 60 = 34560 elements
  // 相比 MHA: 2 * n_h * d_h * l = 2 * 128 * 128 * 60 = 1,966,080 elements
  // MLA KV cache 仅为 MHA 的 1.76%
  ```

  **DeepSeekMoE 算法流程**：
  ```
  Input: u_t ∈ R^d (FFN 输入)
  
  // 1. 共享专家（所有 token 都经过）
  h_shared = sum_{i=1}^{N_s} FFN_i^{(s)}(u_t)   // N_s=2
  
  // 2. 路由专家（Top-K 选择）
  s_{i,t} = Softmax_i(u_t^T @ e_i)              // token-expert affinity, N_r=160
  g_{i,t} = s_{i,t} if s_{i,t} ∈ TopK({s_{j,t}}, K_r=6) else 0
  
  // 3. Device-Limited Routing（约束到最多 M=3 个设备）
  // 先选 M 个 affinity 最高的设备，再在设备内做 Top-K
  
  // 4. 输出
  h_t' = u_t + h_shared + sum_{i=1}^{N_r} g_{i,t} * FFN_i^{(r)}(u_t)
  
  // 辅助损失:
  L_ExpBal = α1 * sum(f_i * P_i)      // expert 级负载均衡, α1=0.003
  L_DevBal = α2 * sum(f_i' * P_i')    // device 级负载均衡, α2=0.05
  L_CommBal = α3 * sum(f_i'' * P_i'') // 通信负载均衡, α3=0.02
  ```
