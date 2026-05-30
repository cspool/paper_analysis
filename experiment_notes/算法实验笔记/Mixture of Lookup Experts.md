## Mixture of Lookup Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MoLE（Mixture of Lookup Experts），一种训练与推理结构不同的 MoE 架构。训练时 routed experts 是 FFN，但以 embedding tokens（embedding 层输出）为输入，且所有 experts 同时激活。推理前将 routed experts 重参数化为 lookup table（LUT），LUT 存储所有 vocabulary 中每个 token 对应的 expert 输出 v_j^i = FFN_j(Embedding(i))，离线预计算后 offload 到存储设备。推理时从 LUT 直接检索 expert 输出，无需计算，仅需一次 lookup + router 加权求和，通信开销可忽略。共享 expert FFN_shared 保持标准计算。
  - 实验比较 Dense baseline（Pythia 架构）、MoE baseline（Mixtral 风格，top-2 routing, 10/34 experts）、MoLE（4/16 experts），在 160M/410M/1B 激活参数规模下。
  - 评估指标：8 个 zero-shot benchmark（ARC-C, ARC-E, BoolQ, HellaSwag, PIQA, RACE, SIQA, LAMBADA）的 accuracy、per-step decoding latency（V100 + HuggingFace Transformers）、#Param Offloaded、#Param Loaded per Token。
  - 消融实验：(a) 训练 loss（LM loss only vs +load_balance vs +z-loss）；(b) routed expert hidden dimension D_r（d/4d/16d）；(c) routed expert 数量 N（2/4/8/16/32）；(d) Architecture Design 逐步演进（MoE-10E → +Full Activation → +Reconfiguration → +Embedding as inputs → +Re-param. = MoLE-4E）；(e) LUT 后训练量化（FP16/NF4/NF3）。

- 硬件平台是什么，配置是什么。
  - 训练硬件：论文未明确说明 GPU 型号。使用 bf16 精度，global batch size=1024，seq length=2048，50000 training iterations。
  - 推理延迟测量：NVIDIA V100 GPU，使用 HuggingFace Transformers。参数加载延迟按 V100 最大 PCIe 带宽 16 GB/s 估算。
  - 训练软件栈：PyTorch + HuggingFace Transformers，基于 Pythia 代码库。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Dense（Pythia 架构）、MoE（Mixtral 风格，无共享 expert，top-2 routing，D_r = Dense FFN hidden_dim / 2）、MoLE（共享 expert = Dense FFN，routed expert D_r = 共享 expert hidden_dim，所有 expert 激活）。具体配置见 Table 2（160M: L=12/d=768/D_s=3072; 410M: L=24/d=1024/D_s=4096; 1B: L=16/d=2048/D_s=8192）。
  - 数据集：100B-token subset of deduped Pile dataset，GPT-NeoX tokenizer（vocab size 50k）。
  - Benchmark：ARC-C, ARC-E, BoolQ, HellaSwag, PIQA, RACE, SIQA, LAMBADA（通过 lm-evaluation-harness 评估，zero-shot accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/JieShibo/MoLE（含 modeling_dense.py, modeling_moe.py, modeling_mole.py, modeling_mole_rep.py）。HuggingFace checkpoint：JieShibo/MoLE-{160M,410M}-{4E,16E}。
  - 算法pipeline 核心流程（训练 → 重参数化 → 推理）：

**训练阶段（MoLE Decoder Layer forward）:**
```
输入: hidden_states ∈ R^{b×s×d}, input_ids ∈ R^{b×s}
      embedding_states = Embedding(input_ids) ∈ R^{b×s×d}

1. Attention:
   residual = hidden_states
   hidden_states = RMSNorm(hidden_states)
   hidden_states = SelfAttention(hidden_states)   // QKV + attention + output proj
   hidden_states = residual + hidden_states

2. Shared Expert (始终激活，接受中间特征):
   residual = hidden_states
   hidden_states = RMSNorm(hidden_states)
   shared_output = FFN_shared(hidden_states)       // [b, s, d_s] → SwiGLU → [b, s, d]

3. Routed Experts (接受 embedding tokens，全激活):
   router_value = SoftMax(Router(hidden_states))   // [b, s, N]
   embedding_states = RMSNorm(embedding_states)
   routed_output = stack([FFN_j(embedding_states) for j in 1..N], dim=2)  // [b, s, N, d]
   routed_output = sum(routed_output * router_value.unsqueeze(-1), dim=2) // [b, s, d]

4. 输出:
   hidden_states = residual + shared_output + routed_output
```
关键差异：routed experts 的输入是 `embedding_states`（仅依赖 input ids），而非 `hidden_states`（中间特征，含上下文）。无 auxiliary loss（因所有 experts 始终激活且可微）。

**重参数化阶段（训练后、推理前）:**
```
# 对每个 expert j 和每个 vocabulary token i，预计算 expert 输出
for j in 1..N:
    for i in 1..|V|:
        e_i = Embedding(i)                         // [d]
        v_j^i = FFN_j(e_i)                         // [d], 只需一次 forward
LUT_l = {v_j^i}_{j=1..N, i=1..|V|}                // size: N × |V| × d
// 实际实现：以 embedding weights 为输入做单次 FFN_j forward
// W_emb ∈ R^{|V|×d} → FFN_j(W_emb) → R^{|V|×d}
```

**推理阶段（MoLE Decoder Layer forward）:**
```
1. Lookup:
   lookup_results = LUT(input_ids)                 // [b, s, N*d]
   lookup_results = lookup_results.view(b, s, N, d)

2. Attention: 同训练

3. Shared Expert: 同训练

4. Routed Expert (计算-free):
   router_value = SoftMax(Router(hidden_states))   // [b, s, N]
   routed_output = sum(lookup_results * router_value.unsqueeze(-1), dim=2) // [b, s, d]

5. 输出: 同训练
```
推理时 routed experts 零 FLOPs，仅 lookup + 加权求和。每 token 加载参数量：dN（仅加载 |V| 中当前 token 对应的 N 个 expert 输出），与 MoE expert offloading 的 2dkD_r 相比，小 1000× 以上。

**复杂度对比（Table 1）:**
- Dense: FLOPs=4dD_s, Offloaded=0, Loaded/token=0
- MoE: FLOPs=4d(kD_r+D_s), Offloaded=2dND_r, Loaded/token=2dkD_r
- MoLE: FLOPs=4dD_s, Offloaded=dN|V|, Loaded/token=dN
